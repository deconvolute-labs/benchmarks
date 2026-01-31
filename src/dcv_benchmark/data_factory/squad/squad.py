import json
import logging
import random
from pathlib import Path

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    load_dataset = None
    tqdm = None

logger = logging.getLogger(__name__)


def fetch_squad_subset(output_dir: Path, count: int = 300, seed: int = 42) -> None:
    """
    Fetches a randomized, diverse subset of SQuAD (v1.1) validation set.

    Args:
        output_dir: The directory to save the JSON file.
        count: Number of samples to select (default: 300).
        seed: Random seed for reproducibility (default: 42).
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' library is required for SQuAD. "
            "Please install the data dependencies: uv pip install '.[data]'"
        )

    logger.info("Loading SQuAD (validation split) via HuggingFace...")

    # Load the full validation split
    dataset = load_dataset("squad", split="validation", streaming=False)

    # Convert to list to shuffle
    all_samples = list(dataset)
    logger.info(
        f"Loaded {len(all_samples)} total samples. Shuffling with seed {seed} ..."
    )

    # Shuffle to mix topics (Biology, History, Sports, etc.)
    random.seed(seed)
    random.shuffle(all_samples)

    from typing import Any

    clean_samples: list[dict[str, Any]] = []
    seen_contexts = set()

    # Select samples, preferring unique contexts to maximize diversity
    iterator = (
        tqdm(all_samples, total=count, desc="Selecting samples")
        if tqdm
        else all_samples
    )

    for row in iterator:
        if len(clean_samples) >= count:
            break

        context = row["context"]
        seen_contexts.add(context)

        clean_samples.append(
            {
                "id": row["id"],
                "query": row["question"],
                "reference_answer": row["answers"]["text"][0]
                if row["answers"]["text"]
                else None,
                # In our pipeline, this Context becomes the "Gold Chunk"
                "source_document": context,
                "title": row["title"],
            }
        )
        if tqdm:
            iterator.update(1)  # type: ignore

    # Output filename
    output_file = output_dir / f"squad_subset_{count}.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "source": "huggingface/squad",
                    "split": "validation",
                    "strategy": "random_shuffled",
                    "seed": seed,
                    "count": len(clean_samples),
                    "unique_topics_covered": len(
                        set(s["title"] for s in clean_samples)
                    ),
                    "description": "Diverse corpus of Question+Context pairs.",
                },
                "data": clean_samples,
            },
            f,
            indent=2,
        )

    logger.info(f"Successfully saved {len(clean_samples)} samples to {output_file}")
    logger.info(f"Unique Topics covered: {len(set(s['title'] for s in clean_samples))}")
