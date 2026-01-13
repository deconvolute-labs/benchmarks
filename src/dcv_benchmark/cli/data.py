import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from dcv_benchmark.data_factory.builder import DatasetBuilder
from dcv_benchmark.data_factory.injector import AttackInjector
from dcv_benchmark.data_factory.loaders import SquadLoader
from dcv_benchmark.models.data_factory import DataFactoryConfig
from dcv_benchmark.utils.logger import get_logger

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


def run_generate_dataset(args: argparse.Namespace) -> None:
    """Handler for the 'data generate' command."""
    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading Data Factory config from {config_path}...")
    config = load_factory_config(config_path)

    try:
        loader = SquadLoader()
        injector = AttackInjector(config=config)

        logger.info(f"Initializing DatasetBuilder for '{config.dataset_name}'...")
        builder = DatasetBuilder(loader=loader, injector=injector, config=config)

        logger.info("Starting build process (Indexing -> Retrieving -> Injecting)...")
        dataset = builder.build()

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Default: dataset.json in the same folder as the config
            output_path = config_path.parent / "dataset.json"

        # safety check
        if output_path.exists() and not args.force:
            logger.error(f"Output file already exists: {output_path}")
            logger.error("Use --force to overwrite it.")
            sys.exit(1)

        logger.info(f"Saving dataset to {output_path}...")
        builder.save(dataset, output_path)
        logger.info("Done.")

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
        "config_path",
        type=str,
        help="Path to the data_factory config (e.g. data/datasets/name/config.yaml)",
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
