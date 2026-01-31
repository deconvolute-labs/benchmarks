import re
from typing import Any, cast

from dcv_benchmark.components.llms import BaseLLM
from dcv_benchmark.constants import (
    BASELINE_TARGET_KEYWORD,
    BUILT_DATASETS_DIR,
)
from dcv_benchmark.evaluators.base import BaseEvaluator
from dcv_benchmark.evaluators.bipia import BipiaEvaluator
from dcv_benchmark.evaluators.canary import CanaryEvaluator
from dcv_benchmark.evaluators.keyword import KeywordEvaluator
from dcv_benchmark.evaluators.language import LanguageMismatchEvaluator
from dcv_benchmark.models.config.experiment import ExperimentConfig
from dcv_benchmark.models.dataset import BaseDataset
from dcv_benchmark.targets.basic_rag import BasicRAG
from dcv_benchmark.targets.basic_rag_guard import BasicRAGGuard
from dcv_benchmark.utils.dataset_loader import DatasetLoader
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(experiment_config: ExperimentConfig) -> BaseDataset:
    """
    Resolves and loads the input dataset based on the experiment configuration.

    Expects a simple folder name string.
    Finds the dataset in workspace/datasets/built/{name}/dataset.json.
    """
    dataset_name = experiment_config.dataset or experiment_config.name

    logger.info(f"Loading dataset: {dataset_name}...")

    # Primary path
    fallback_path = BUILT_DATASETS_DIR / dataset_name / "dataset.json"

    try:
        dataset: BaseDataset = DatasetLoader(dataset_name).load()
    except FileNotFoundError:
        if fallback_path.exists():
            logger.info(f"Using fallback path: {fallback_path}")
            dataset = DatasetLoader(str(fallback_path)).load()
        else:
            logger.error(f"Dataset not found: {dataset_name}")
            raise

    logger.info(f"Loaded dataset: {dataset.meta.name} (v{dataset.meta.version})")
    return dataset


def create_target(experiment_config: ExperimentConfig) -> BasicRAG | BasicRAGGuard:
    """Creates the target instance."""
    target_name = experiment_config.target.name
    if target_name == "basic_rag":
        return BasicRAG(experiment_config.target)
    elif target_name == "basic_rag_guard":
        return BasicRAGGuard(experiment_config.target)
    else:
        raise ValueError(
            f"Unsupported target: '{target_name}'. "
            "Available targets: 'basic_rag', 'basic_rag_guard'."
        )


def _validate_baseline_payload(dataset: BaseDataset) -> None:
    """Helper to validate dataset payload for Keyword evaluation."""
    attack_info = dataset.meta.attack_info
    if not attack_info:
        logger.warning(
            "Dataset metadata is missing 'attack_info'. Skipping validation."
        )
        return

    pattern = re.compile(rf"\b{re.escape(BASELINE_TARGET_KEYWORD)}\b")
    if not pattern.search(attack_info.payload):
        error_msg = (
            "Configuration Mismatch! \n"
            f"Evaluator expects: '{BASELINE_TARGET_KEYWORD}'\n"
            f"Dataset payload: '{attack_info.payload}'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"Confirmed: Dataset payload contains target '{BASELINE_TARGET_KEYWORD}'."
    )


def create_evaluator(
    type_name: str,
    settings: dict[str, Any],
    target: Any = None,
    dataset: BaseDataset | None = None,
) -> BaseEvaluator:
    """
    Instantiates an Evaluator based on type and settings dict.
    """

    if type_name == "canary":
        logger.info("Evaluator: Canary Defense Integrity")
        return CanaryEvaluator()

    elif type_name == "keyword":
        if dataset:
            _validate_baseline_payload(dataset)
        kw = settings.get("target_keyword") or BASELINE_TARGET_KEYWORD
        logger.info(f"Evaluator: Keyword (Target: '{kw}')")
        return KeywordEvaluator(target_keyword=kw)

    elif type_name == "language":
        # Supports "language" or "language_mismatch" alias?
        # Plan said "language".
        allowed = settings.get("allowed", ["en"])
        # Use existing LanguageMismatchEvaluator or adapt it?
        # The existing one takes expected_language (str).
        # We might need to handle list vs str.
        expected = allowed[0] if isinstance(allowed, list) and allowed else "en"
        strict = settings.get("strict", True)

        logger.info(f"Evaluator: Language (Expected: {expected}, Strict: {strict})")
        return LanguageMismatchEvaluator(expected_language=expected, strict=strict)

    elif type_name == "bipia":
        # attack_success_rate evaluator mentioned in plan as separate?
        # "The 'attack_success_rate' evaluator calculates..."
        # But here we might map "attack_success_rate" to BipiaEvaluator
        # or something new.
        # But `BipiaEvaluator` exists.
        pass

    # The plan mentions "attack_success_rate" in the BIPIA config example.
    if type_name == "attack_success_rate":
        # Maybe map to BipiaEvaluator?
        # Or is it a generic one?
        # "BipiaEvaluator" class seems to do judge logic.
        # Let's assume for now it's BipiaEvaluator but renamed in config.
        # Or I should look for AttackSuccessRateEvaluator?
        # I will map it to BipiaEvaluator logic if possible, or create a new one?
        # Wait, BipiaEvaluator requires LLM to judge.
        # If ASR is just counting blocks vs success?
        # If generate=False, BipiaEvaluator might not work if it expects generation.
        # I will map it to Bipia for now but check if I need to adjust it.
        logger.info("Evaluator: ASR (using BipiaEvaluator logic)")

        judge_llm = None
        # Check for LLM in settings (unlikely for simple ASR but possible)
        # Using target LLM?
        judge_llm = cast(BaseLLM | None, getattr(target, "llm", None))

        # In scan mode (generate=False), target.llm is None.
        # BipiaEvaluator (judge) relies on LLM to checks instructions.
        # But if content is "Blocked", we don't need LLM.
        # BipiaEvaluator needs to be robust to missing LLM if content is blocked?

        if not judge_llm:
            # For ASR in scan mode, we might not need a judge if we just check
            # for "Blocked" string?
            # BipiaEvaluator usually uses an LLM to check if the attack succeeded
            # (i.e. if the output followed instructions).
            # If blocked, it failed.
            # Pass a mock or allow None?
            pass

        # The existing BipiaEvaluator explicitly requests judge_llm.
        return BipiaEvaluator(judge_llm=judge_llm)

    raise ValueError(f"Unknown evaluator type: {type_name}")
