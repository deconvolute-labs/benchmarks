from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from dcv_benchmark.analytics.calculators.security import SecurityMetricsCalculator
from dcv_benchmark.analytics.plotter import Plotter
from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.models.metrics import SecurityMetrics
from dcv_benchmark.models.report import ExperimentReport, ReportMeta
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


RESULTS_ARTIFACT_FILENAME = "results.json"
TRACES_FILENAME = "traces.jsonl"


class ReportGenerator:
    """
    Assembles the final results artifact for an experiment run.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.results_path = run_dir / RESULTS_ARTIFACT_FILENAME
        self.traces_path = run_dir / TRACES_FILENAME
        self.plotter = Plotter(run_dir)

    def generate(
        self, config: ExperimentConfig, start_time: datetime, end_time: datetime
    ) -> Path:
        """
        Orchestrates the creation of the final report.

        1. Selects the appropriate MetricsCalculator based on the experiment type.
        2. Computes metrics from the raw traces.jsonl file.
        3. Combines Metadata, Config, and Metrics into a single dictionary.
        4. Writes the result to disk.

        Args:
            config: The full configuration object used for the run.
            start_time: Timestamp when the run started.
            end_time: Timestamp when the run finished.

        Returns:
            The Path to the generated results file.
        """

        duration = (end_time - start_time).total_seconds()

        calculator = SecurityMetricsCalculator()
        logger.info("Calculating metrics from traces...")

        metrics_data: SecurityMetrics = calculator.calculate(self.traces_path)
        logger.info("Generating plots...")
        self.plotter.generate_all(metrics_data)

        # Assemble the report structure
        report = ExperimentReport(
            meta=ReportMeta(
                name=config.name,
                description=config.description,
                timestamp_start=start_time.replace(microsecond=0),
                timestamp_end=end_time.replace(microsecond=0),
                duration_seconds=round(duration, 2),
                deconvolute_version=version("deconvolute"),
            ),
            config=config.model_dump(),
            metrics=metrics_data,
        )

        with open(self.results_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        logger.info(f"Report saved to: {self.results_path}")
        return self.results_path
