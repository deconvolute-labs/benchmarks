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


def download_data(source: str, output_dir: str | None = None) -> None:
    """
    Fetches raw datasets (SQuAD, BIPIA) to the workspace.
    """
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Default: workspace/datasets/raw/{source}
        output_path = RAW_DATASETS_DIR / source

    logger.info(f"Preparing to download '{source}' data to {output_path}...")

    try:
        if source == "squad":
            download_squad(output_path)
        elif source == "bipia":
            download_bipia(output_path)
        else:
            logger.error(f"Unknown source: '{source}'. Options: squad, bipia")
            sys.exit(1)

        logger.info(f"Download of '{source}' complete.")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


def build_data(
    config_path_str: str, name: str | None = None, overwrite: bool = False
) -> None:
    """
    Generates (injects/builds) a dataset from a recipe config.
    """
    config_path = Path(config_path_str)

    # 1. Resolve Config Path
    # If directory provided, look for dataset_config.yaml
    if config_path.is_dir():
        potential = config_path / "dataset_config.yaml"
        if potential.exists():
            config_path = potential
        else:
            logger.error(
                "Directory provided but 'dataset_config.yaml' not found in "
                f"{config_path}"
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
    if name:
        dataset_name = name
        # Update config to match the build name so metadata is consistent
        config.dataset_name = dataset_name
    else:
        dataset_name = config.dataset_name

    target_dir = BUILT_DATASETS_DIR / dataset_name

    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / "dataset.json"

    # 4. Check Overwrite
    if output_file.exists():
        if not overwrite:
            logger.error(f"Dataset artifact '{output_file}' already exists.")
            logger.info("Use --overwrite to replace it.")
            sys.exit(1)
        else:
            logger.warning(f"Overwriting existing dataset artifact at {output_file}...")
            output_file.unlink()

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
        builder.save(dataset, output_file)

        logger.info(f"Build successful! Artifacts saved to: {target_dir}")

    except Exception as e:
        logger.exception(f"Build failed: {e}")
        # Cleanup partial build - only if we created the file and it failed?
        if output_file.exists() and not overwrite:
            output_file.unlink()
        sys.exit(1)
