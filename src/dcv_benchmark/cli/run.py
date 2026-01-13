import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dcv_benchmark.runner import ExperimentRunner
from dcv_benchmark.utils.experiment_loader import load_experiment
from dcv_benchmark.utils.logger import get_logger, print_experiment_header, setup_logger

logger = get_logger(__name__)


def resolve_config_path(user_input: str) -> Path:
    """
    Smartly resolves the experiment config path.
    1. Checks path exactly as given.
    2. Checks inside 'experiments/' directory.
    3. Checks with '.yaml' suffix.
    """
    path = Path(user_input)
    if path.exists():
        return path

    experiments_dir = Path("experiments")
    if (experiments_dir / path).exists():
        return experiments_dir / path

    if not path.suffix:
        if path.with_suffix(".yaml").exists():
            return path.with_suffix(".yaml")
        if (experiments_dir / path.with_suffix(".yaml")).exists():
            return experiments_dir / path.with_suffix(".yaml")

    return path


def run_experiment_command(args: argparse.Namespace) -> None:
    """Handler for the 'run' command (executing benchmarks)."""
    # Initialize logger for the run
    setup_logger(level="DEBUG" if args.debug else "INFO")

    try:
        config_path = resolve_config_path(args.config_name)
        logger.debug(f"Resolved experiment path: {config_path}")

        logger.debug(f"Loading experiment from {config_path}...")
        experiment_config = load_experiment(config_path)

        print_experiment_header(experiment_config.model_dump())
        logger.info(f"Loaded experiment from {config_path}...")

        if experiment_config.target.defense.required_version:
            installed_version = version("deconvolute")
            if installed_version != experiment_config.target.defense.required_version:
                raise ImportError(
                    "Deconvolute version mismatch. "
                    f"Required: {experiment_config.target.defense.required_version}, "
                    f"Found: {installed_version}"
                )

        if args.dry_run:
            logger.info("Initializing Target for dry-run validation...")
            # Late import to avoid heavy dependencies if not needed
            from dcv_benchmark.targets.basic_rag import BasicRAG

            # We assume BasicRAG is the target for now, or dispatch based on config
            BasicRAG(experiment_config.target)
            logger.info(
                f"Dry run successful. Experiment '{experiment_config.name}' is valid."
            )
            return

        # Execute workload
        logger.info("Initializing Orchestrator...")
        runner = ExperimentRunner()

        output_path = runner.run(experiment_config, limit=args.limit)
        logger.info(f"Artifacts saved to: {output_path}")

    except Exception as e:
        if args.debug:
            logger.exception("Fatal error during execution")
        else:
            logger.error(f"Fatal error: {e}")
        sys.exit(1)


def register_run_cli(subparsers: Any) -> None:
    """Registers the 'run' command."""
    run_parser = subparsers.add_parser("run", help="Execute a benchmark experiment.")
    run_parser.add_argument(
        "config_name",
        type=str,
        help="Name of the experiment config file (e.g. 'generator_canary.yaml').",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and components without executing the workload.",
    )
    run_parser.add_argument(
        "--limit", type=int, help="Limit execution to N samples (useful for debugging)."
    )
    run_parser.set_defaults(func=run_experiment_command)
