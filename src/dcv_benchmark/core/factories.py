from typing import Any, cast

from dcv_benchmark.components.llms import BaseLLM
from dcv_benchmark.constants import (
    BUILT_DATASETS_DIR,
)
from dcv_benchmark.evaluators.base import BaseEvaluator
from dcv_benchmark.evaluators.bipia import BipiaDefenseEvaluator
from dcv_benchmark.evaluators.squad import SquadDefenseEvaluator
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


def create_experiment_evaluators(
    experiment_config: ExperimentConfig, target: Any, dataset: BaseDataset
) -> dict[str, BaseEvaluator]:
    """
    Automatically selects the CORRECT evaluator suite based on the dataset type.
    Manual selection is forbidden to prevent misconfiguration.
    """
    evaluators: dict[str, BaseEvaluator] = {}

    # 1. SQuAD Logic
    if dataset.meta.type == "squad":
        logger.info("Configuration: Detected SQuAD. Using 'SquadDefenseEvaluator'.")
        evaluators["squad_defense"] = SquadDefenseEvaluator(
            target_config=experiment_config.target, dataset=dataset
        )
        return evaluators

    # 2. BIPIA Logic
    if dataset.meta.type == "bipia":
        logger.info("Configuration: Detected BIPIA. Using 'BipiaDefenseEvaluator'.")
        # For BIPIA, we generally need the LLM to judge.
        judge_llm = cast(BaseLLM | None, getattr(target, "llm", None))
        evaluators["bipia_asr"] = BipiaDefenseEvaluator(judge_llm=judge_llm)
        return evaluators

    # Fallback / Warning
    logger.warning(
        f"No automated evaluators defined for dataset type: {dataset.meta.type}"
    )
    return evaluators
