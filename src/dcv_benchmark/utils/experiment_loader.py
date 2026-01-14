from pathlib import Path

import yaml
from pydantic import ValidationError

from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


def load_experiment(path: Path) -> ExperimentConfig:
    """
    Loads, parses, and validates the experiment configuration YAML.

    Args:
        path: Path to the .yaml file defining the experiment.

    Returns:
        A validated 'ExperimentConfig' object.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file is invalid YAML or fails schema validation.
    """

    if not path.exists():
        logger.error(f"Experiment file not found at: {path}")
        raise FileNotFoundError(f"Experiment file not found: {path}")

    try:
        with open(path, encoding="utf-8") as file_handler:
            raw_data = yaml.safe_load(file_handler)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML: {e}")
        raise ValueError(f"Failed to parse YAML file: {e}") from e

    if not raw_data or "experiment" not in raw_data:
        raise ValueError(
            f"Invalid experiment file at {path}: Missing top-level 'experiment' key."
        )

    try:
        # Validate against the Pydantic Schema
        experiment = ExperimentConfig(**raw_data["experiment"])
        logger.debug(
            f"Experiment '{experiment.name}' loaded and validated successfully."
        )
        return experiment

    except ValidationError as e:
        logger.error(f"Experiment validation failed: {e}")
        raise ValueError(f"Invalid experiment configuration: {e}") from e
