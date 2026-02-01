import datetime
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


def _center_text(text: str, width: int = 90) -> str:
    """Helper to center text within the standard width."""
    return f"{text}".center(width)


def print_experiment_header(config: dict[str, Any]) -> None:
    """
    Logs a standardized visual header for the experiment startup.
    """
    logger = get_logger(__name__)

    name = config.get("name", "Unnamed Experiment")
    raw_version = config.get("version", "N/A")
    # Remove 'v' prefix if present for cleaner display
    version = raw_version.lstrip("v") if isinstance(raw_version, str) else raw_version
    desc = config.get("description", "")

    logger.info("=" * 90)
    logger.info(_center_text("DECONVOLUTE BENCHMARK"))
    logger.info("=" * 90)
    logger.info(f"Experiment     : {name}")
    logger.info(f"Version        : {version}")
    logger.info(f"DCV SDK        : {dcv_version}")
    if desc:
        logger.info(f"Description    : {desc}")
    logger.info("=" * 90)


def print_dataset_header(meta: Any) -> None:
    """
    Prints a formatted header for the loaded dataset.
    Accepts a DatasetMetadata object or a dict.
    """
    logger = get_logger(__name__)

    # Handle Pydantic model or dict
    if hasattr(meta, "model_dump"):
        data = meta.model_dump()
    else:
        data = meta if isinstance(meta, dict) else {}

    name = data.get("name", "Unnamed Dataset")
    version = data.get("version", "")

    # Attack Info is optional
    attack_info = data.get("attack_info")
    if attack_info:
        strategy = attack_info.get("strategy", "Unknown")
        rate = attack_info.get("rate", 0.0)
        # Convert rate to percentage string
        rate_str = f"{rate * 100:.0f}%"
    else:
        strategy = None
        rate_str = None

    logger.info("")
    logger.info("=" * 90)
    logger.info(_center_text(f"DATASET: {name} (version {version})"))
    logger.info("-" * 90)

    if strategy:
        logger.info(f"Strategy       : {strategy.upper()}")
        logger.info(f"Injection Rate : {rate_str}")
    else:
        logger.info("Type           : Benign / Validation Only")

    logger.info("=" * 90)
    logger.info("")


def print_run_summary(metrics: Any, duration: float, artifacts_path: str) -> None:
    """
    Logs the final summary statistics of a benchmark run.
    Expects a GlobalSecurityMetrics object.
    """
    logger = get_logger(__name__)

    # metrics is GlobalSecurityMetrics
    total = metrics.total_samples

    # Determine Status
    # We consider it a 'Pass' if the system behaved as expected (High PNA, Low ASR)
    # But for the summary, we just show the stats.

    logger.info("=" * 90)
    logger.info(_center_text("RUN COMPLETE"))
    logger.info("=" * 90)

    # 1. High Level Stats
    logger.info(f"Duration       : {duration:.2f}s")
    logger.info(f"Total Samples  : {total}")
    logger.info(f"Avg Latency    : {metrics.avg_latency_seconds:.4f}s")
    logger.info("-" * 90)

    # 2. Security Metrics (The core KPIs)
    # ASR: Attack Success Rate (Lower is better)
    # PNA: Performance on No Attack (Higher is better)
    logger.info(
        f"ASR (Attack Success Rate)      : {metrics.asr_score:.2%}  (Lower is better)"
    )
    logger.info(
        f"PNA (Benign Accuracy)          : {metrics.pna_score:.2%}  (Higher is better)"
    )
    logger.info("-" * 90)

    # 3. Confusion Matrix Breakdown
    # TP: Attacks Caught | FN: Attacks Missed
    # TN: Benign Allowed | FP: Benign Blocked
    logger.info(f"Attacks Caught (TP)            : {metrics.tp}")
    logger.info(f"Attacks Missed (FN)            : {metrics.fn}")
    logger.info(f"Benign Allowed (TN)            : {metrics.tn}")
    logger.info(f"False Positives (FP)           : {metrics.fp}")
    logger.info("=" * 90)
    logger.info(f"Artifacts: {artifacts_path}")
    logger.info("=" * 90)


class ExperimentProgressLogger:
    """
    Handles logging of experiment progress, including start messages,
    step updates, and ETA calculations.
    """

    def __init__(self, total_samples: int):
        self.total_samples: int = total_samples
        self.start_time: datetime.datetime | None = None
        self.logger: logging.Logger = get_logger(__name__)
        # interval for logging progress (10%)
        self.log_interval = max(1, self.total_samples // 10)

    def start(self) -> None:
        """
        Logs the start of the experiment.
        """

        self.start_time = datetime.datetime.now()
        self.logger.info(
            f"🚀 [STARTED] Experiment started with {self.total_samples} samples."
        )

    def log_progress(self, current_count: int, success_count: int) -> None:
        """
        Logs progress if the current count hits the 10% interval or is the last sample.
        Calculates ETA if the elapsed time is sufficient.
        """

        # Check if we should log (10% interval or last sample)
        if (current_count) % self.log_interval == 0 or (
            current_count
        ) == self.total_samples:
            if self.start_time is None:
                self.start_time = datetime.datetime.now()

            pct = (current_count / self.total_samples) * 100
            elapsed = datetime.datetime.now() - self.start_time

            # success rate calculation
            if current_count > 0:
                success_rate = (success_count / current_count) * 100
            else:
                success_rate = 0.0

            msg = (
                f"🔄 [RUNNING] Progress: {current_count}/{self.total_samples} "
                f"({pct:.0f}%) | Success Rate: {success_rate:.1f}%"
            )

            # ETA Calculation
            # Only show ETA if we are past the first interval and it's taking some time
            # This avoids ETA on super fast runs
            seconds_elapsed = elapsed.total_seconds()
            if seconds_elapsed > 5 and current_count < self.total_samples:
                avg_time_per_sample = seconds_elapsed / current_count
                remaining_samples = self.total_samples - current_count
                eta_seconds = remaining_samples * avg_time_per_sample

                # Format ETA
                if eta_seconds < 60:
                    eta_str = "< 1 min"
                else:
                    eta_min = int(eta_seconds // 60)
                    eta_str = f"~{eta_min} min"

                msg += f" | ETA: {eta_str}"

            self.logger.info(msg)
