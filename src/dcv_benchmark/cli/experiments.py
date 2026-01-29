import argparse
import sys
from pathlib import Path

import yaml

from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.runner import ExperimentRunner
from dcv_benchmark.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def handle_run(args: argparse.Namespace) -> None:
    """
    Handles the 'experiment run' command.
    Loads the config, validates it, and triggers the runner.
    """
    config_path = Path(args.config)
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
    if args.debug_traces:
        setup_logging("DEBUG")
        logger.debug("Debug traces enabled.")

    logger.info(f"Starting Experiment: {exp_config.name} (v{exp_config.version})")

    # 3. Initialize Runner
    # Results will be saved next to the config file by default
    output_dir = config_path.parent / "results"
    runner = ExperimentRunner(output_dir=output_dir)

    try:
        # 4. Execute
        runner.run(exp_config, limit=args.limit)
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        sys.exit(1)


def register_experiment_commands(subparsers) -> None:
    """Registers the 'experiment' subcommand group."""
    exp_parser = subparsers.add_parser("experiment", help="Experiment execution tools")
    exp_subs = exp_parser.add_subparsers(dest="experiment_command", required=True)

    # Run Command
    run_parser = exp_subs.add_parser(
        "run", help="Execute an experiment from a config file"
    )
    run_parser.add_argument(
        "config", help="Path to the experiment.yaml configuration file"
    )
    run_parser.add_argument(
        "--limit", type=int, help="Limit execution to N samples (for debugging)"
    )
    run_parser.add_argument(
        "--debug-traces",
        action="store_true",
        help="Enable verbose logging and full-text traces",
    )
    run_parser.set_defaults(func=handle_run)
