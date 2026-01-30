import sys
from pathlib import Path

import yaml

from dcv_benchmark.core.runner import ExperimentRunner
from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def run_experiment(
    config_path_str: str, limit: int | None = None, debug_traces: bool = False
) -> None:
    """
    Loads the config, validates it, and triggers the runner.
    """
    config_path = Path(config_path_str)
    if not config_path.exists():
        logger.error(f"Experiment config file not found: {config_path}")
        sys.exit(1)

    # 1. Load Configuration
    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # We expect the config to be under an 'experiment' key
        if "experiment" not in raw_config:
            logger.error("Invalid config format: Missing top-level 'experiment' key.")
            sys.exit(1)

        exp_config = ExperimentConfig(**raw_config["experiment"])
    except Exception as e:
        logger.error(f"Failed to parse experiment config: {e}")
        sys.exit(1)

    # 2. Setup Logging
    # If the user requested debug traces, we might want to adjust log levels
    if debug_traces:
        # Note: calling setup_logging here might override global settings.
        # But if the user passed --debug-traces specific to this command,
        # it makes sense.
        setup_logging("DEBUG")
        logger.debug("Debug traces enabled.")

    logger.info(f"Starting Experiment: {exp_config.name} (v{exp_config.version})")

    # 3. Initialize Runner
    # Results will be saved next to the config file by default
    output_dir = config_path.parent / "results"
    runner = ExperimentRunner(output_dir=output_dir)

    try:
        # 4. Execute
        runner.run(exp_config, limit=limit, debug_traces=debug_traces)
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        sys.exit(1)
