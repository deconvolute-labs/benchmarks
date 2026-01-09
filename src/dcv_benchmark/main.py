import argparse
import sys
from pathlib import Path

from dcv_benchmark.runner import ExperimentRunner
from dcv_benchmark.utils.experiment_loader import load_experiment
from dcv_benchmark.utils.logger import print_experiment_header, setup_logger


def resolve_config_path(user_input: str) -> Path:
    """
    Smartly resolves the config path.
    1. Checks if the path exists exactly as given.
    2. Checks if it exists inside the 'experiments/' directory.
    3. Checks if it exists with '.yaml' appended (if missing).

    Args:
        user_input: The user-provided path.

    Returns:
        The path to the experiment config file.
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

    # Return original to allow FileNotFoundError to be raised with the user's input
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deconvolute Benchmark: Security Evaluation Harness"
    )
    parser.add_argument(
        "config_name",
        type=str,
        help=(
            "Name of the config file (e.g. 'generator_canary.yaml'). "
            "Directory 'experiments/' is optional."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and components without executing the workload.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--limit", type=int, help="Limit execution to N samples (useful for debugging)."
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(level=log_level)

    try:
        config_path = resolve_config_path(args.config_name)
        logger.debug(f"Resolved experiment path: {config_path}")

        logger.debug(f"Loading experiment from {config_path}...")
        experiment_config = load_experiment(config_path)

        print_experiment_header(experiment_config.model_dump())
        logger.info(f"Loaded experiment from {config_path}...")

        if args.dry_run:
            logger.info("Initializing Target for dry-run validation...")
            from dcv_benchmark.targets.basic_rag import BasicRAG

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


if __name__ == "__main__":
    main()
