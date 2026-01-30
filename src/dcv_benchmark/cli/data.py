import sys
from pathlib import Path
from typing import Any

import yaml

from dcv_benchmark.constants import BUILT_DATASETS_DIR, RAW_DATASETS_DIR
from dcv_benchmark.data_factory.bipia.bipia import BipiaBuilder
from dcv_benchmark.data_factory.downloader import download_bipia, download_squad
from dcv_benchmark.data_factory.injector import AttackInjector
from dcv_benchmark.data_factory.loaders import SquadLoader
from dcv_benchmark.data_factory.squad.squad_builder import SquadBuilder
from dcv_benchmark.models.bipia_config import BipiaConfig
from dcv_benchmark.models.data_factory import DataFactoryConfig
from dcv_benchmark.models.dataset import Dataset, DatasetMeta
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

    # If directory provided, look for dataset_config.yaml
    if config_path.is_dir():
        potential = config_path / "dataset_config.yaml"
        if potential.exists():
            config_path = potential
        else:
            # Fallback for BIPIA naming convention if user put it there
            potential_bipia = config_path / "bipia_config.yaml"
            if potential_bipia.exists():
                config_path = potential_bipia
            else:
                logger.error(f"No config file found in {config_path}")
                sys.exit(1)

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # 1. Load Raw YAML to determine type
    try:
        with open(config_path, encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to parse YAML: {e}")
        sys.exit(1)

    # 2. Branch Logic
    is_bipia = "tasks" in raw_yaml  # 'tasks' is unique to BipiaConfig

    if is_bipia:
        _build_bipia(raw_yaml, name, overwrite)
    else:
        _build_squad(raw_yaml, name, overwrite)


def _build_bipia(raw_config: dict[str, Any], name: str | None, overwrite: bool) -> None:
    """Handler for BIPIA datasets."""
    try:
        config = BipiaConfig(**raw_config)
    except Exception as e:
        logger.error(f"Invalid BIPIA config: {e}")
        sys.exit(1)

    dataset_name = name or config.dataset_name
    target_dir = BUILT_DATASETS_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / "dataset.json"

    if output_file.exists() and not overwrite:
        logger.error(f"Dataset '{output_file}' exists. Use --overwrite.")
        sys.exit(1)

    logger.info(f"Building BIPIA dataset '{dataset_name}'...")

    try:
        # Initialize Builder
        builder = BipiaBuilder(raw_dir=RAW_DATASETS_DIR / "bipia", seed=config.seed)

        # Build Samples
        raw_samples = builder.build(
            tasks=config.tasks,
            injection_pos=config.injection_pos,
            max_samples=config.max_samples,
        )

        # Wrap in Standard Dataset Object for compatibility with Runner
        dataset = Dataset(
            meta=DatasetMeta(
                name=dataset_name,
                version="1.0.0",
                description=(
                    f"BIPIA Benchmark (Tasks: {config.tasks}, Pos: "
                    f"{config.injection_pos})"
                ),
                author="Deconvolute / Microsoft BIPIA",
            ),
            samples=raw_samples,
        )

        # Save
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=2))

        logger.info(
            f"Build successful! Saved {len(raw_samples)} samples to: {output_file}"
        )
        # Save config copy for reproducibility
        with open(target_dir / "bipia_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(raw_config, f)

    except Exception as e:
        logger.exception(f"BIPIA Build failed: {e}")
        sys.exit(1)


# TODO: REname to squad dataest
def _build_squad(raw_config: dict[str, Any], name: str | None, overwrite: bool) -> None:
    """Handler for Standard (SQuAD/Canary) datasets."""
    try:
        config = DataFactoryConfig(**raw_config)
    except Exception as e:
        logger.error(f"Invalid Standard config: {e}")
        sys.exit(1)

    dataset_name = name or config.dataset_name
    target_dir = BUILT_DATASETS_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / "dataset.json"

    if output_file.exists() and not overwrite:
        logger.error(f"Dataset '{output_file}' exists. Use --overwrite.")
        sys.exit(1)

    logger.info(f"Building Standard dataset '{dataset_name}'...")

    try:
        loader = SquadLoader()
        injector = AttackInjector(config)
        builder = SquadBuilder(loader=loader, injector=injector, config=config)
        dataset = builder.build()
        builder.save(dataset, output_file)

        logger.info(f"Build successful! Artifacts saved to: {target_dir}")

    except Exception as e:
        logger.exception(f"Standard Build failed: {e}")
        sys.exit(1)
