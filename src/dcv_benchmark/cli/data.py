import argparse
import shutil
import sys
from pathlib import Path

import yaml

from dcv_benchmark.constants import BUILT_DATASETS_DIR, RAW_DATASETS_DIR
from dcv_benchmark.data_factory.builder import DatasetBuilder
from dcv_benchmark.data_factory.downloader import download_bipia, download_squad
from dcv_benchmark.data_factory.injector import AttackInjector
from dcv_benchmark.data_factory.loaders import SquadLoader
from dcv_benchmark.models.data_factory import DataFactoryConfig
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


def handle_download(args: argparse.Namespace) -> None:
    """
    Handles the 'data download' command.
    Fetches raw datasets (SQuAD, BIPIA) to the workspace.
    """
    source = args.source

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: workspace/datasets/raw/{source}
        output_dir = RAW_DATASETS_DIR / source

    logger.info(f"Preparing to download '{source}' data to {output_dir}...")

    try:
        if source == "squad":
            download_squad(output_dir)
        elif source == "bipia":
            download_bipia(output_dir)
        else:
            logger.error(f"Unknown source: '{source}'. Options: squad, bipia")
            sys.exit(1)

        logger.info(f"Download of '{source}' complete.")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


def handle_build(args: argparse.Namespace) -> None:
    """
    Handles the 'data build' command.
    Generates (injects/builds) a dataset from a recipe config.
    """
    config_path = Path(args.config)

    # 1. Resolve Config Path
    # If directory provided, look for dataset_config.yaml
    if config_path.is_dir():
        potential = config_path / "dataset_config.yaml"
        if potential.exists():
            config_path = potential
        else:
            logger.error(
                f"Directory provided but 'dataset_config.yaml' not found in {config_path}"
            )
            sys.exit(1)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # 2. Load Config
    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        config = DataFactoryConfig(**raw_config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # 3. Determine Dataset Name (CLI override > Config > Folder Name)
    if args.name:
        dataset_name = args.name
        # Update config to match the build name so metadata is consistent
        config.dataset_name = dataset_name
    else:
        dataset_name = config.dataset_name

    target_dir = BUILT_DATASETS_DIR / dataset_name

    # 4. Check Overwrite
    if target_dir.exists():
        if not args.overwrite:
            logger.error(f"Dataset '{dataset_name}' already exists at {target_dir}.")
            logger.info("Use --overwrite to replace it.")
            sys.exit(1)
        else:
            logger.warning(f"Overwriting existing dataset at {target_dir}...")
            shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build Dataset
    logger.info(f"Building dataset '{dataset_name}' from {config_path}...")

    try:
        # Note: We default to SquadLoader as it handles the JSON format.
        # Future loaders (e.g. BIPIA direct loader) can be selected here based on
        # config if needed.
        loader = SquadLoader()
        injector = AttackInjector(config)
        builder = DatasetBuilder(loader=loader, injector=injector, config=config)

        dataset = builder.build()

        # 6. Save Artifacts
        output_file = target_dir / "dataset.json"
        builder.save(dataset, output_file)

        # Save a copy of the config for reproducibility
        shutil.copy(config_path, target_dir / "dataset_config.yaml")

        logger.info(f"Build successful! Artifacts saved to: {target_dir}")

    except Exception as e:
        logger.exception(f"Build failed: {e}")
        # Cleanup partial build
        if target_dir.exists() and not args.overwrite:
            shutil.rmtree(target_dir)
        sys.exit(1)


def register_data_commands(subparsers) -> None:
    """Registers the 'data' subcommand group."""
    data_parser = subparsers.add_parser("data", help="Data Factory tools")
    data_subs = data_parser.add_subparsers(dest="data_command", required=True)

    # --- Download Command ---
    dl_parser = data_subs.add_parser(
        "download", help="Fetch raw datasets (SQuAD, BIPIA)"
    )
    dl_parser.add_argument(
        "source",
        choices=["squad", "bipia"],
        help="Name of the source dataset to download",
    )
    dl_parser.add_argument(
        "--output-dir",
        help="Override default output directory (workspace/datasets/raw/...)",
    )
    dl_parser.set_defaults(func=handle_download)

    # --- Build Command ---
    build_parser = data_subs.add_parser(
        "build", help="Generate/Inject a dataset from a recipe"
    )
    build_parser.add_argument(
        "config", help="Path to the dataset configuration file (YAML)"
    )
    build_parser.add_argument(
        "--name", help="Name for the built dataset (overrides config name)"
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset if it exists",
    )
    build_parser.set_defaults(func=handle_build)
