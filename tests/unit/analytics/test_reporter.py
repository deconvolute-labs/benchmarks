import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from dcv_benchmark.analytics.reporter import (
    RESULTS_ARTIFACT_FILENAME,
    TRACES_FILENAME,
    ReportGenerator,
)
from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.models.metrics import GlobalSecurityMetrics, SecurityMetrics


@pytest.fixture
def mock_config():
    # Minimal config for the report
    return ExperimentConfig(
        name="test_run",
        description="A test run",
        dataset="squad_val",
        target={
            "name": "rag",
            "defense": {
                "ingestion": {},
                "generation": {
                    "canary_detector": {"enabled": True, "settings": {}},
                    "language_detector": {"enabled": False, "settings": {}},
                    "prompt_guard": {"enabled": False},
                },
            },
            "system_prompt": {"file": "p.yaml", "key": "k"},
            "prompt_template": {"file": "t.yaml", "key": "k"},
            "pipeline_params": {},
        },
        evaluators={},
    )


@pytest.fixture
def mock_metrics():
    return SecurityMetrics(
        global_metrics=GlobalSecurityMetrics(
            total_samples=10,
            pna_score=1.0,
            asr_score=0.0,
            fpr_score=0.0,
            tp=0,
            fn=0,
            tn=10,
            fp=0,
            avg_latency_seconds=0.1,
            latencies_attack=[],
            latencies_benign=[],
        ),
        by_strategy={},
    )


class TestReportGenerator:
    @patch("dcv_benchmark.analytics.reporter.SecurityMetricsCalculator")
    @patch("dcv_benchmark.analytics.reporter.Plotter")
    def test_generate_success(
        self, MockPlotter, MockCalculator, mock_config, mock_metrics, tmp_path
    ):
        reporter = ReportGenerator(tmp_path)
        # Setup mocks
        mock_calc_instance = MockCalculator.return_value
        mock_calc_instance.calculate.return_value = mock_metrics

        mock_plotter_instance = MockPlotter.return_value

        start_time = datetime(2023, 1, 1, 12, 0, 0)
        end_time = start_time + timedelta(seconds=10)

        # Ensure traces file path is what we expect
        expected_traces_path = tmp_path / TRACES_FILENAME

        # Call generate
        result_path = reporter.generate(mock_config, start_time, end_time)

        # Verify calculator was called with correct path
        mock_calc_instance.calculate.assert_called_once_with(expected_traces_path)

        # Verify plotter was called
        mock_plotter_instance.generate_all.assert_called_once_with(mock_metrics)

        # Verify file creation
        assert result_path == tmp_path / RESULTS_ARTIFACT_FILENAME
        assert result_path.exists()

        # Verify content of the report
        with open(result_path) as f:
            data = json.load(f)

        assert data["meta"]["name"] == "test_run"
        assert data["meta"]["duration_seconds"] == 10.0
        assert data["metrics"]["global_metrics"]["total_samples"] == 10

    @patch("dcv_benchmark.analytics.reporter.SecurityMetricsCalculator")
    @patch("dcv_benchmark.analytics.reporter.Plotter")
    def test_generate_calculation_failure(
        self, MockPlotter, MockCalculator, mock_config, tmp_path
    ):
        reporter = ReportGenerator(tmp_path)
        # Setup mock to raise exception
        mock_calc_instance = MockCalculator.return_value
        mock_calc_instance.calculate.side_effect = ValueError("Calculation failed")

        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)

        # Call generate and expect exception
        with pytest.raises(ValueError, match="Calculation failed"):
            reporter.generate(mock_config, start_time, end_time)

        # Plotter should NOT be called if calculation fails
        MockPlotter.return_value.generate_all.assert_not_called()

        # Verify result file is NOT created
        result_path = tmp_path / RESULTS_ARTIFACT_FILENAME
        assert not result_path.exists()
