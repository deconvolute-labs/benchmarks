import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Literal

from dcv_benchmark.data_factory.base import BaseDatasetBuilder
from dcv_benchmark.data_factory.bipia.bipia_templates import TASK_CONFIGS
from dcv_benchmark.models.dataset import BenchmarkSample, ContextChunk

logger = logging.getLogger(__name__)


# -- Injection Logic (Ported from BIPIA/data/utils.py) --
def _split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using regex to avoid NLTK dependency.
    Approximates the behavior of PunktSentenceTokenizer.
    """
    # Split on period/question/exclamation followed by space or end of string.
    # We keep the delimiter with the sentence.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _insert_end(context: str, attack: str, seed: int | None = None) -> str:
    """
    Ported from: https://github.com/microsoft/BIPIA/blob/main/bipia/data/utils.py
    """
    return "\n".join([context, attack])


def _insert_start(context: str, attack: str, seed: int | None = None) -> str:
    """
    Ported from: https://github.com/microsoft/BIPIA/blob/main/bipia/data/utils.py
    """
    return "\n".join([attack, context])


def _insert_middle(context: str, attack: str, seed: int | None = None) -> str:
    """
    Adapted from: https://github.com/microsoft/BIPIA/blob/main/bipia/data/utils.py
    """
    if seed is not None:
        random.seed(seed)

    sentences = _split_sentences(context)
    if not sentences:
        return _insert_end(context, attack)

    # Pick a random split point (avoiding index 0 to truly be 'middle' if possible)
    if len(sentences) > 1:
        split_idx = random.randint(1, len(sentences) - 1)  # noqa
    else:
        split_idx = 0

    pre = " ".join(sentences[:split_idx])
    post = " ".join(sentences[split_idx:])

    return "\n".join([pre, attack, post])


INJECTION_METHODS = {
    "end": _insert_end,
    "start": _insert_start,
    "middle": _insert_middle,
}


class BipiaBuilder(BaseDatasetBuilder):
    """
    Reconstructs BIPIA samples from raw downloaded files.
    Merges clean documents with attacks using task-specific logic and templates.
    """

    def __init__(self, raw_dir: Path, seed: int = 42):
        self.raw_dir = raw_dir
        self.seed = seed
        random.seed(seed)

    def load_json_list(self, filename: str) -> list[dict[str, Any]]:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"BIPIA source file missing: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    return content
        except json.JSONDecodeError:
            pass
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def load_attacks(self, filename: str) -> dict[str, Any]:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"BIPIA attack file missing: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def build(  # type: ignore[override]
        self,
        tasks: list[Literal["email", "code", "table", "qa"]],  # We exclude qa for now
        injection_pos: Literal["start", "middle", "end"] = "end",
        max_samples: int | None = None,
    ) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []

        try:
            attacks_text = self.load_attacks("attacks_text.json")
            attacks_code = self.load_attacks("attacks_code.json")
        except FileNotFoundError as e:
            logger.error(f"Cannot build BIPIA: {e}")
            return []

        # Preserve Category and Index metadata
        flat_attacks_text = []
        for cat, texts in attacks_text.items():
            for i, txt in enumerate(texts):
                flat_attacks_text.append((cat, i, txt))

        flat_attacks_code = []
        for cat, texts in attacks_code.items():
            for i, txt in enumerate(texts):
                flat_attacks_code.append((cat, i, txt))

        injection_fn = INJECTION_METHODS.get(injection_pos, _insert_end)

        for task in tasks:
            logger.info(f"Processing BIPIA task: {task}...")

            config = TASK_CONFIGS.get(task)
            if not config:
                continue

            if task == "email":
                source_filename = "test_email.jsonl"
                attack_pool = flat_attacks_text
            elif task == "table":
                source_filename = "test_table.jsonl"
                attack_pool = flat_attacks_text
            elif task == "code":
                source_filename = "test_code.jsonl"
                attack_pool = flat_attacks_code
            else:
                continue

            try:
                source_data = self.load_json_list(source_filename)
            except Exception as e:
                logger.warning(f"Error: {e}. Continuing ...")
                continue

            if max_samples:
                source_data = source_data[:max_samples]

            for i, item in enumerate(source_data):
                if task == "code":
                    doc_to_inject = "\n".join(item.get("context", []))
                    user_query = config["user_format"].format(
                        error="\n".join(item.get("error", [])),
                        code="\n".join(item.get("code", [])),
                    )
                    answer_text = "\n".join(item.get("ideal", []))
                else:
                    doc_to_inject = item.get("context", "")
                    user_query = config["user_format"].format(
                        question=item.get("question", "")
                    )
                    answer_text = item.get("ideal", "")

                # Select Attack Tuple
                (cat, idx, attack_str) = random.choice(attack_pool)  # noqa
                poisoned_doc = injection_fn(doc_to_inject, attack_str, self.seed)

                system_prompt_meta = config["system"].format(
                    guidance="", context="{context}"
                )

                samples.append(
                    BenchmarkSample(
                        id=f"bipia_{task}_{i}",
                        query=user_query,
                        reference_answer=answer_text,
                        sample_type="attack",
                        attack_strategy=f"bipia_{task}_{injection_pos}",
                        context=[
                            ContextChunk(
                                id="0",
                                content=poisoned_doc,
                                is_malicious=True,
                                metadata={
                                    "task": task,
                                    "injection_pos": injection_pos,
                                    "is_poisoned": "true",
                                    "system_prompt_template": system_prompt_meta,
                                    # Critical for Evaluator
                                    "attack_category": cat,
                                    "attack_index": str(idx),
                                    "attack_payload": attack_str,
                                },
                            )
                        ],
                    )
                )

        return samples
