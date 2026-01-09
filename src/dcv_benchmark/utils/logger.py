import logging
import sys
from typing import Any

from deconvolute import __version__ as dcv_version


def setup_logger(name: str = "dcv_benchmark", level: str = "INFO") -> logging.Logger:
    """
    Configures and returns the centralized logger.

    Args:
        name: The name of the logger (default: 'dcv_benchmark').
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # Prevent adding duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger

    # Console Handler (Stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())

    formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "dcv_benchmark") -> logging.Logger:
    """
    Helper to get the configured logger instance in other modules.
    """
    return logging.getLogger(name)


def print_experiment_header(config: dict[str, Any]) -> None:
    """
    Logs a standardized visual header for the experiment startup.
    """
    logger = get_logger()

    name = config.get("name", "Unnamed Experiment")
    version = config.get("version", "N/A")
    desc = config.get("description", "")

    # A visual separator block
    logger.info("=" * 65)
    logger.info("DECONVOLUTE BENCHMARK")
    logger.info("=" * 65)
    logger.info(f"Experiment:      {name}")
    logger.info(f"Version:         {version}")
    logger.info(f"DCV SDK version: {dcv_version}")
    if desc:
        logger.info(f"Description:     {desc}")
    logger.info("=" * 65)


def print_run_summary(
    total: int, success: int, duration: Any, artifacts_path: str
) -> None:
    """
    Logs the final summary statistics of a benchmark run.
    """
    logger = get_logger()
    failed = total - success
    pass_rate = (success / total * 100) if total > 0 else 0.0

    logger.info("=" * 65)
    logger.info("RUN COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Total Samples:  {total}")
    logger.info(f"Passed:         {success}")
    logger.info(f"Failed:         {failed}")
    logger.info(f"Pass Rate:      {pass_rate:.1f}%")
    logger.info(f"Duration:       {duration}")
    logger.info(f"Artifacts:      {artifacts_path}")
    logger.info("=" * 65)
