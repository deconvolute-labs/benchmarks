import logging
import sys
from typing import Any

from deconvolute import __version__ as dcv_version


class CustomFormatter(logging.Formatter):
    """
    Formatter that shows the logger name only in DEBUG mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self.debug_formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"
        )
        self.default_formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.DEBUG:
            return self.debug_formatter.format(record)
        return self.default_formatter.format(record)


def setup_logging(level: str | int = "INFO") -> None:
    """
    Configures the root logger with a standardized format.

    Sets up a stream handler that prints to stdout. It uses a custom formatter
    that includes the logger name only when in DEBUG mode, keeping INFO logs clean.
    It also silences noisy third-party libraries (like `httpx` and `chromadb`).

    Args:
        level (str | int): The desired logging level (e.g. "DEBUG", "INFO").
            Defaults to "INFO".

    Note:
        This function uses `force=True`, meaning it will overwrite any existing
        logging configuration. This is intentional to ensure consistent formatting
        during benchmark runs.
    """
    # Convert string level to int if necessary
    if isinstance(level, str):
        level = level.upper()

    # Create handler with custom formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())

    # Basic configuration for the root logger
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,  # Overwrite any existing config (useful for testing/notebooks)
    )

    # Quiet down some noisy 3rd party libraries if they exist
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the given name.
    """
    return logging.getLogger(name)


def print_experiment_header(config: dict[str, Any]) -> None:
    """
    Logs a standardized visual header for the experiment startup.
    """
    logger = get_logger(__name__)

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
    logger = get_logger(__name__)
    failed = total - success
    pass_rate = (success / total * 100) if total > 0 else 0.0

    logger.info("=" * 90)
    logger.info("RUN COMPLETE")
    logger.info("=" * 90)
    logger.info(f"Total Samples:  {total}")
    logger.info(f"Passed:         {success}")
    logger.info(f"Failed:         {failed}")
    logger.info(f"Pass Rate:      {pass_rate:.1f}%")
    logger.info(f"Duration:       {duration}")
    logger.info(f"Artifacts:      {artifacts_path}")
    logger.info("=" * 90)


def print_dataset_header(config: dict[str, Any]) -> None:
    """Prints a formatted header for the dataset generation."""
    # We expect a DataFactoryConfig dumped as dict
    name = config.get("dataset_name", "Unnamed Dataset")
    strategy = config.get("attack_strategy", "Unknown")
    corpus = config.get("source_file", "N/A")
    rate = config.get("attack_rate", 0.0)

    logger = get_logger(__name__)

    logger.info("")
    logger.info("=" * 90)
    logger.info(f"DATASET GENERATION: {name}")
    logger.info("-" * 90)
    logger.info(f"Corpus    : {corpus}")
    logger.info(f"Strategy  : {strategy.upper()}")
    logger.info(f"Inj. Rate : {rate * 100:.0f}%")
    logger.info("=" * 90)
    logger.info("")
