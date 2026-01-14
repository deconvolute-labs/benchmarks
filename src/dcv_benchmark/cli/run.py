import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dcv_benchmark.cli.data import generate_dataset, load_factory_config
from dcv_benchmark.constants import SCENARIOS_DIR
from dcv_benchmark.runner import ExperimentRunner
from dcv_benchmark.utils.experiment_loader import load_experiment
from dcv_benchmark.utils.logger import (
    get_logger,
    print_experiment_header,
    setup_logger,
)

logger = get_logger(__name__)


def parse_scenario_argument(arg: str) -> tuple[str, str]:
    """
    Parses 'scenario_name' or 'scenario_name:variant'.
    Returns (scenario_name, variant_suffix).
    Example: 'canary:gpt4' -> ('canary', 'gpt4')
             'canary'      -> ('canary', '')
    """
    if ":" in arg:
        parts = arg.split(":")
        return parts[0], parts[1]
    return arg, ""


def resolve_scenario_paths(scenario_name: str, variant: str) -> Path:
    """
    Locates the experiment config file within the scenarios directory.
    """
    scenario_dir = SCENARIOS_DIR / scenario_name

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    # Determine filename: experiment.yaml or experiment_<variant>.yaml
    if variant:
        filename = f"experiment_{variant}.yaml"
    else:
        filename = "experiment.yaml"

    config_path = scenario_dir / filename

    if not config_path.exists():
        msg = (
            f"Experiment config not found: {config_path}\n"
            f"Expected a file named '{filename}' in '{scenario_dir}'."
        )

        # Suggest colon syntax if they tried the default
        if not variant:
            msg += (
                "\nTip: To run a variant (e.g. 'experiment_baseline.yaml'), "
                "use the syntax 'scenario:baseline'."
            )

        # List what IS there
        existing_yamls = [f.name for f in scenario_dir.glob("*.yaml")]
        if existing_yamls:
            msg += f"\nAvailable configs: {', '.join(existing_yamls)}"

        raise FileNotFoundError(msg)

    return config_path


def ensure_dataset_exists(experiment_config: Any, config_path: Path) -> None:
    """
    Checks if the dataset referenced in the experiment exists.
    If missing attempts to generate it using 'dataset_config.yaml' from the same folder.
    """
    # Resolve the absolute path of the dataset
    # We assume the path in the config is relative to the config file location
    # if it's not an absolute path.
    raw_path_str = experiment_config.input.dataset_path

    if raw_path_str:
        # User specified a custom path/name
        raw_path = Path(raw_path_str)
        if raw_path.is_absolute():
            dataset_path = raw_path
        else:
            dataset_path = config_path.parent / raw_path
    else:
        # DEFAULT: No path provided -> assume "dataset.json" in scenario folder
        dataset_path = config_path.parent / "dataset.json"

    # Update the config object with the absolute path so the runner finds it
    experiment_config.input.dataset_path = str(dataset_path)

    if dataset_path.exists():
        logger.debug(f"Dataset found at: {dataset_path}")
        return

    # Dataset missing - Try to generate
    logger.warning(
        f"Dataset not found at {dataset_path}. Attempting lazy generation..."
    )

    scenario_dir = config_path.parent
    dataset_config_path = scenario_dir / "dataset_config.yaml"

    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"Dataset is missing and cannot be generated.\n"
            f"Expected 'dataset_config.yaml' at {dataset_config_path}"
        )

    logger.info(f"Generating dataset from {dataset_config_path}...")

    # Generation Logic
    factory_config = load_factory_config(dataset_config_path)
    generate_dataset(factory_config, dataset_path)


def run_experiment_command(args: argparse.Namespace) -> None:
    """Handler for the 'run' command."""
    setup_logger(level="DEBUG" if args.debug else "INFO")

    try:
        # Resolve Paths
        scenario_name, variant = parse_scenario_argument(args.target)
        config_path = resolve_scenario_paths(scenario_name, variant)

        scenario_dir = config_path.parent

        logger.debug(f"Resolved experiment: {config_path}")

        # Load Config
        experiment_config = load_experiment(config_path)

        print_experiment_header(experiment_config.model_dump())

        # Lazy Data Generation
        if not args.dry_run:
            ensure_dataset_exists(experiment_config, config_path)

        # Version Checks
        if experiment_config.target.defense.required_version:
            installed_version = version("deconvolute")
            if installed_version != experiment_config.target.defense.required_version:
                logger.warning(
                    "Deconvolute version mismatch. "
                    f"Required: {experiment_config.target.defense.required_version}, "
                    f"Found: {installed_version}"
                )

        if args.dry_run:
            logger.info("Dry run successful. Config is valid.")
            return

        # Execution
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = scenario_dir / "results"

        runner = ExperimentRunner(output_dir=output_dir)
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
        "target",
        type=str,
        help=(
            "Target scenario in format 'name' or 'name:variant' "
            "(e.g. 'canary_naive' or 'canary_naive:gpt4')."
        ),
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and components without executing the workload.",
    )
    run_parser.add_argument(
        "--limit", type=int, help="Limit execution to N samples (useful for debugging)."
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        help=(
            "Directory to save experiment results. "
            "Defaults to 'scenarios/<name>/results'."
        ),
    )
    run_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    run_parser.set_defaults(func=run_experiment_command)
