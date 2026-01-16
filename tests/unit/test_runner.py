from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.constants import BASELINE_TARGET_KEYWORD
from dcv_benchmark.models.experiments_config import (
    DefenseConfig,
    DefenseLayerConfig,
    EvaluatorConfig,
    ExperimentConfig,
    InputConfig,
    ScenarioConfig,
    TargetConfig,
)
from dcv_benchmark.runner import ExperimentRunner


@pytest.fixture
def mock_dataset_loader():
    with patch("dcv_benchmark.runner.DatasetLoader") as loader:
        mock_ds = MagicMock()
        mock_ds.samples = [MagicMock(id=f"s{i}") for i in range(5)]
        mock_ds.meta.attack_info.payload = (
            f"some payload with {BASELINE_TARGET_KEYWORD}"
        )
        loader.return_value.load.return_value = mock_ds
        yield loader


@pytest.fixture
def valid_config():
    return ExperimentConfig(
        name="unit_test_exp",
        description="unit test",
        input=InputConfig(dataset_path="dummy.json"),
        target=TargetConfig(
            name="basic_rag",
            defense=DefenseConfig(
                type="deconvolute",
                layers=[DefenseLayerConfig(type="canary", enabled=True, settings={})],
            ),
            # Minimal other fields to pass validation
            system_prompt={"file": "s", "key": "k"},
            prompt_template={"file": "p", "key": "k"},
        ),
        scenario=ScenarioConfig(id="test"),
        evaluator=EvaluatorConfig(
            type="keyword", target_keyword=BASELINE_TARGET_KEYWORD
        ),
    )


def test_init_creates_dir(tmp_path):
    output_dir = tmp_path / "custom_results"
    _ = ExperimentRunner(output_dir=output_dir)
    assert output_dir.exists()


def test_run_missing_dataset_path(valid_config, tmp_path):
    runner = ExperimentRunner(output_dir=tmp_path)
    valid_config.input.dataset_path = None

    with pytest.raises(ValueError, match="Cannot find path to dataset"):
        runner.run(valid_config)


def test_run_missing_evaluator(valid_config, tmp_path):
    runner = ExperimentRunner(output_dir=tmp_path)
    valid_config.evaluator = None

    with (
        patch("dcv_benchmark.runner.DatasetLoader"),
        patch("dcv_benchmark.runner.BasicRAG"),
    ):
        with pytest.raises(ValueError, match="No evaluator specified"):
            runner.run(valid_config)


def test_validate_baseline_payload_mismatch(tmp_path):
    """Should raise ValueError if dataset payload doesn't contain target keyword."""
    runner = ExperimentRunner(output_dir=tmp_path)

    mock_dataset = MagicMock()
    mock_dataset.meta.attack_info.payload = "innocent text"

    # We temporarily rely on the private helper logic or integration flow
    # Easier to call private method directly for unit testing logic
    with pytest.raises(ValueError, match="Configuration Mismatch"):
        runner._validate_baseline_payload(mock_dataset)


@patch("dcv_benchmark.runner.BasicRAG")
@patch("dcv_benchmark.runner.KeywordEvaluator")
@patch("dcv_benchmark.runner.ReportGenerator")
def test_run_with_limit(
    MockReport, MockKeyword, MockRAG, mock_dataset_loader, valid_config, tmp_path
):
    """Verify processing stops after limit is reached."""
    runner = ExperimentRunner(output_dir=tmp_path)

    # Dataset has 5 samples (from fixture)
    # Set limit to 2
    runner.run(valid_config, limit=2)

    # Verify BasicRAG invoke called exactly 2 times
    assert MockRAG.return_value.invoke.call_count == 2


@patch("dcv_benchmark.runner.BasicRAG")
@patch("dcv_benchmark.runner.KeywordEvaluator")
@patch("dcv_benchmark.runner.ReportGenerator")
def test_run_handles_exception_single_sample(
    MockReport, MockKeyword, MockRAG, mock_dataset_loader, valid_config, tmp_path
):
    """Experiment should continue even if one sample crashes."""
    runner = ExperimentRunner(output_dir=tmp_path)

    # Make BasicRAG raise error on first call, succeed on second
    instance = MockRAG.return_value
    instance.invoke.side_effect = [Exception("Crash"), MagicMock()]

    runner.run(valid_config, limit=2)

    # Should have attempted both (or up to limit if we didn't crash entirely)
    assert instance.invoke.call_count == 2
    # Verify report generated implies run finished
    MockReport.return_value.generate.assert_called_once()
