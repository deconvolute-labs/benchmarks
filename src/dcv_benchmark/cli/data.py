import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from dcv_benchmark.constants import BUILT_DATASETS_DIR
from dcv_benchmark.data_factory.builder import DatasetBuilder
from dcv_benchmark.data_factory.injector import AttackInjector
from dcv_benchmark.data_factory.loaders import SquadLoader
from dcv_benchmark.models.data_factory import DataFactoryConfig
from dcv_benchmark.utils.logger import get_logger, print_dataset_header

logger = get_logger(__name__)


def load_factory_config(path: Path) -> DataFactoryConfig:
    """Helper to load and validate the Data Factory YAML config."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return DataFactoryConfig(**raw)
    except Exception as e:
        logger.error(f"Failed to parse config file {path}: {e}")
        sys.exit(1)


def resolve_data_target(target: str) -> Path:
    """
    Resolves the 'target' argument to a config file path.
    1. Checks if 'target' is a dataset name (folder in BUILT_DATASETS_DIR).
       If so, looks for 'dataset_config.yaml' inside it.
    2. Checks if 'target' is a direct file path.
    """
    # Try Dataset Name
    dataset_dir = BUILT_DATASETS_DIR / target
    if dataset_dir.exists() and dataset_dir.is_dir():
        # It's a valid dataset folder
        config_candidate = dataset_dir / "dataset_config.yaml"
        if config_candidate.exists():
            return config_candidate

    # Try File Path
    path = Path(target)
    if path.exists():
        return path

    # Error Handling
    if dataset_dir.exists() and dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder '{target}' found, but missing 'dataset_config.yaml' "
            f"at {dataset_dir / 'dataset_config.yaml'}"
        )

    raise FileNotFoundError(
        f"Target not found. Checked dataset folder '{dataset_dir}' and path '{path}'."
    )


def generate_dataset(config: DataFactoryConfig, output_path: Path) -> None:
    """
    Core logic to generate and save a dataset based on the provided config.
    """
    print_dataset_header(config.model_dump())

    loader = SquadLoader()
    injector = AttackInjector(config=config)

    logger.debug(f"Initializing DatasetBuilder for '{config.dataset_name}'...")
    builder = DatasetBuilder(loader=loader, injector=injector, config=config)

    logger.info("Starting build process (Indexing -> Retrieving -> Injecting)...")
    dataset = builder.build()

    logger.info(f"Saving dataset to {output_path}...")
    builder.save(dataset, output_path)
    logger.info("Done.")


def run_generate_dataset(args: argparse.Namespace) -> None:
    """Handler for the 'data generate' command."""
    try:
        config_path = resolve_data_target(args.target)
        logger.debug(f"Resolved dataset config: {config_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.debug(f"Loading Data Factory config from {config_path}...")
    config = load_factory_config(config_path)

    try:
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Default: dataset.json in the same folder as the config
            output_path = config_path.parent / "dataset.json"

        # Safety check
        if output_path.exists() and not args.force:
            logger.error(f"Output file already exists: {output_path}")
            logger.error("Use --force to overwrite it.")
            sys.exit(1)

        generate_dataset(config, output_path)

    except Exception:
        logger.exception("Fatal error during dataset generation")
        sys.exit(1)


def register_data_cli(subparsers: Any) -> None:
    """Registers the 'data' command group."""
    # Create 'data' parent command
    data_parser = subparsers.add_parser("data", help="Data Factory commands")
    data_subs = data_parser.add_subparsers(dest="data_command", required=True)

    # Create 'generate' subcommand
    gen_parser = data_subs.add_parser(
        "generate", help="Generate a synthetic RAG dataset from a config file."
    )
    gen_parser.add_argument(
        "target",
        type=str,
        help="Scenario name (e.g. 'canary_naive') or path to config file.",
    )
    gen_parser.add_argument(
        "-o", "--output", type=str, help="Custom output path for the JSON file."
    )
    gen_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )

    # Map this command to the handler function
    gen_parser.set_defaults(func=run_generate_dataset)
