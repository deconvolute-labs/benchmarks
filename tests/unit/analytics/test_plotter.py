from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.analytics.plotter import PLOT_DIR, Plotter
from dcv_benchmark.models.metrics import (
    GlobalSecurityMetrics,
    SecurityMetrics,
    StrategySecurityMetric,
)


@pytest.fixture
def mock_metrics():
    return SecurityMetrics(
        global_metrics=GlobalSecurityMetrics(
            total_samples=100,
            pna_score=0.9,
            asv_score=0.1,
            fpr_score=0.0,
            tp=10,
            fn=5,
            tn=80,
            fp=5,
            avg_latency_seconds=0.5,
            latencies_attack=[0.5, 0.6],
            latencies_benign=[0.1, 0.2],
        ),
        by_strategy={
            "strat1": StrategySecurityMetric(
                samples=10, asv=0.2, detected_count=8, missed_count=2
            ),
            "strat2": StrategySecurityMetric(
                samples=5, asv=0.0, detected_count=5, missed_count=0
            ),
        },
    )


@pytest.fixture
def plotter(tmp_path):
    return Plotter(tmp_path)


def test_plotter_init_creates_dir(tmp_path):
    Plotter(tmp_path)
    assert (tmp_path / PLOT_DIR).exists()
    assert (tmp_path / PLOT_DIR).is_dir()


@patch("dcv_benchmark.analytics.plotter.plt")
def test_generate_all(mock_plt, plotter, mock_metrics):
    # Mocking standard plot calls
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    plotter.generate_all(mock_metrics)

    # Check if subplots was called at least 3 times (once for each plot)
    assert mock_plt.subplots.call_count == 3
    # Check if savefig was called 3 times
    assert mock_plt.savefig.call_count == 3
    # Check if close was called 3 times
    assert mock_plt.close.call_count == 3


@patch("dcv_benchmark.analytics.plotter.plt")
def test_plot_confusion_matrix(mock_plt, plotter, mock_metrics):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    plotter._plot_confusion_matrix(mock_metrics)

    # Verify imshow called with correct matrix shape
    args, _ = mock_ax.imshow.call_args
    matrix = args[0]
    assert matrix.shape == (2, 2)
    # Check values: [[TP, FN], [FP, TN]]
    assert matrix[0, 0] == 10
    assert matrix[0, 1] == 5
    assert matrix[1, 0] == 5
    assert matrix[1, 1] == 80

    mock_plt.savefig.assert_called_once()
    assert "confusion_matrix.png" in str(mock_plt.savefig.call_args[0][0])


@patch("dcv_benchmark.analytics.plotter.plt")
def test_plot_strategy_asv(mock_plt, plotter, mock_metrics):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    plotter._plot_strategy_asv(mock_metrics)

    mock_ax.barh.assert_called_once()
    args, _ = mock_ax.barh.call_args
    # Check y_pos length
    assert len(args[0]) == 2
    # Check values (asv scores)
    assert len(args[1]) == 2

    mock_plt.savefig.assert_called_once()
    assert "asv_by_strategy.png" in str(mock_plt.savefig.call_args[0][0])


@patch("dcv_benchmark.analytics.plotter.plt")
def test_plot_latency_distribution(mock_plt, plotter, mock_metrics):
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    plotter._plot_latency_distribution(mock_metrics)

    # Should call hist twice (once for benign, once for attack)
    assert mock_ax.hist.call_count == 2

    mock_plt.savefig.assert_called_once()
    assert "latency_distribution.png" in str(mock_plt.savefig.call_args[0][0])


@patch("dcv_benchmark.analytics.plotter.plt")
def test_plot_strategy_asv_empty(mock_plt, plotter):
    empty_metrics = SecurityMetrics(
        global_metrics=GlobalSecurityMetrics(
            total_samples=0,
            pna_score=0,
            asv_score=0,
            fpr_score=0.0,
            tp=0,
            fn=0,
            tn=0,
            fp=0,
            avg_latency_seconds=0,
            latencies_attack=[],
            latencies_benign=[],
        ),
        by_strategy={},
    )

    # Should simply return without plotting
    plotter._plot_strategy_asv(empty_metrics)

    mock_plt.subplots.assert_not_called()
    mock_plt.savefig.assert_not_called()


@patch("dcv_benchmark.analytics.plotter.plt")
def test_plot_latency_empty(mock_plt, plotter):
    empty_metrics = SecurityMetrics(
        global_metrics=GlobalSecurityMetrics(
            total_samples=0,
            pna_score=0,
            asv_score=0,
            fpr_score=0.0,
            tp=0,
            fn=0,
            tn=0,
            fp=0,
            avg_latency_seconds=0,
            latencies_attack=[],
            latencies_benign=[],
        ),
        by_strategy={},
    )

    plotter._plot_latency_distribution(empty_metrics)

    mock_plt.subplots.assert_not_called()
    mock_plt.savefig.assert_not_called()
