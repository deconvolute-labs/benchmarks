import re
from typing import Any, cast

from dcv_benchmark.components.llms import BaseLLM, create_llm
from dcv_benchmark.constants import (
    AVAILABLE_EVALUATORS,
    BASELINE_TARGET_KEYWORD,
    BUILT_DATASETS_DIR,
    RAW_DATASETS_DIR,
)
from dcv_benchmark.data_factory.bipia.bipia import BipiaBuilder
from dcv_benchmark.evaluators.base import BaseEvaluator
from dcv_benchmark.evaluators.bipia import BipiaEvaluator
from dcv_benchmark.evaluators.canary import CanaryEvaluator
from dcv_benchmark.evaluators.keyword import KeywordEvaluator
from dcv_benchmark.evaluators.language import LanguageMismatchEvaluator
from dcv_benchmark.models.config.experiment import EvaluatorConfig, ExperimentConfig
from dcv_benchmark.models.dataset import BaseDataset, BipiaDataset, DatasetMeta
from dcv_benchmark.targets.basic_rag import BasicRAG
from dcv_benchmark.targets.basic_rag_guard import BasicRAGGuard
from dcv_benchmark.utils.dataset_loader import DatasetLoader
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(experiment_config: ExperimentConfig) -> BaseDataset:
    """Loads (or builds) dataset based on config."""
    input_config = experiment_config.input

    # -- Case 1: BIPIA (On-the-fly build) --
    if input_config.type == "bipia":
        logger.info("Building BIPIA dataset in-memory...")
        builder = BipiaBuilder(
            raw_dir=RAW_DATASETS_DIR / "bipia", seed=input_config.seed
        )
        samples = builder.build(
            tasks=input_config.tasks,
            injection_pos=input_config.injection_pos,
            max_samples=input_config.max_samples,
        )

        # Wrap in ephemeral BipiaDataset
        dataset = BipiaDataset(
            meta=DatasetMeta(
                name=f"bipia_ephemeral_{experiment_config.name}",
                type="bipia",
                version="1.0.0-mem",
                description="Ephemeral BIPIA dataset built from config",
                author="Deconvolute Labs (Runtime)",
            ),
            samples=samples,
        )
        logger.info(f"Built BIPIA dataset with {len(samples)} samples.")
        return dataset

    # -- Case 2: SQuAD / Standard (Load from disk) --
    elif input_config.type == "squad":
        # input_config is SquadInputConfig
        dataset_name = input_config.dataset_name
        if not dataset_name:
            # Fallback: Use Experiment Name
            logger.info(
                "No dataset name in config. Attempting fallback to experiment name."
            )
            dataset_name = experiment_config.name

        fallback_path = BUILT_DATASETS_DIR / dataset_name / "dataset.json"

        # Try loading via loader (which handles resolution)
        try:
            dataset: BaseDataset = DatasetLoader(dataset_name).load()  # type: ignore
        except FileNotFoundError:
            # Retry with direct fallback path to be helpful
            if fallback_path.exists():
                logger.info(f"Using fallback path: {fallback_path}")
                dataset = DatasetLoader(str(fallback_path)).load()  # type: ignore
            else:
                raise

        logger.info(f"Loaded dataset: {dataset.meta.name} (v{dataset.meta.version})")
        logger.info(f"Description: {dataset.meta.description}")
        return dataset

    else:
        raise ValueError(f"Unknown input config type: {input_config.type}")


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

        judge_llm: BaseLLM | None = None

        # Priority 1: Use explicit evaluator LLM config
        if config.llm:
            logger.info("Using explicit LLM config for BIPIA Judge.")
            judge_llm = create_llm(config.llm)

        # Priority 2: Fallback to Target's LLM (if valid type)
        else:
            logger.info(
                "No explicit evaluator LLM. Attempting fallback to Target's LLM."
            )
            judge_llm = cast(BaseLLM | None, getattr(target, "llm", None))

        if not judge_llm:
            error_msg = (
                "BIPIA Evaluator requires a Judge LLM! "
                "Please provide 'llm' in evaluator config or "
                "ensure target has an accessible 'llm' attribute."
            )
            logger.error(error_msg)
            # We strictly enforce LLM presence now as requested
            raise ValueError(error_msg)

        return BipiaEvaluator(judge_llm=judge_llm)
    else:
        raise ValueError(f"Unknown evaluator type: {config.type}")
