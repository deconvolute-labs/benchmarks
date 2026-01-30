import re
from typing import Any

from dcv_benchmark.constants import (
    AVAILABLE_EVALUATORS,
    BASELINE_TARGET_KEYWORD,
    BUILT_DATASETS_DIR,
)
from dcv_benchmark.evaluators.base import BaseEvaluator
from dcv_benchmark.evaluators.bipia import BipiaEvaluator
from dcv_benchmark.evaluators.canary import CanaryEvaluator
from dcv_benchmark.evaluators.keyword import KeywordEvaluator
from dcv_benchmark.evaluators.language import LanguageMismatchEvaluator
from dcv_benchmark.models.config.experiment import EvaluatorConfig, ExperimentConfig
from dcv_benchmark.models.dataset import BaseDataset
from dcv_benchmark.targets.basic_rag import BasicRAG
from dcv_benchmark.targets.basic_rag_guard import BasicRAGGuard
from dcv_benchmark.utils.dataset_loader import DatasetLoader
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(experiment_config: ExperimentConfig) -> BaseDataset:
    """Loads dataset based on config or default path."""
    dataset_path_or_name = experiment_config.input.dataset_name
    if not dataset_path_or_name:
        fallback_path = BUILT_DATASETS_DIR / experiment_config.name / "dataset.json"
        if not fallback_path.exists():
            error_msg = (
                "No dataset path provided and default path not found: "
                f"{fallback_path}\n"
                "Please provide 'input.dataset_name' in config or ensure the "
                "default dataset exists."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"No dataset provided. Using default path: {fallback_path}")
        dataset_path_or_name = str(fallback_path)

    dataset: BaseDataset = DatasetLoader(dataset_path_or_name).load()
    logger.info(f"Loaded dataset: {dataset.meta.name} (v{dataset.meta.version})")
    logger.info(f"Description: {dataset.meta.description}")
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
    config: EvaluatorConfig | None,
    target: Any = None,
    dataset: BaseDataset | None = None,
) -> BaseEvaluator:
    """Creates the evaluator instance."""
    if config is None:
        error_msg = (
            "Missing Configuration: No evaluator specified.\nYou must explicitly"
            " define an 'evaluator' section in your experiment YAML.\n"
            f"Available types: {', '.join(AVAILABLE_EVALUATORS)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if config.type == "canary":
        logger.info("Evaluator: Canary Defense Integrity")
        return CanaryEvaluator()

    elif config.type == "keyword":
        if dataset:
            _validate_baseline_payload(dataset)
        kw = config.target_keyword or BASELINE_TARGET_KEYWORD
        logger.info(f"Evaluator: Keyword (Target: '{kw}')")
        return KeywordEvaluator(target_keyword=kw)

    elif config.type == "language_mismatch":
        logger.info(
            f"Evaluator: Language Mismatch (Expected: {config.expected_language})"
        )
        try:
            return LanguageMismatchEvaluator(
                expected_language=config.expected_language,
                strict=config.strict,
            )
        except ImportError as e:
            logger.error("Missing dependencies for Language Evaluator.")
            raise e
    elif config.type == "bipia":
        logger.info("Evaluator: BIPIA (LLM Judge + Pattern Match)")
        judge_llm = getattr(target, "llm", None)
        if not judge_llm:
            logger.warning(
                "BIPIA Evaluator initialized without an LLM! Text tasks will fail."
            )
        return BipiaEvaluator(judge_llm=judge_llm)
    else:
        raise ValueError(f"Unknown evaluator type: {config.type}")
