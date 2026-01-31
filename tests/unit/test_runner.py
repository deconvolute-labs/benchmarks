from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.constants import BASELINE_TARGET_KEYWORD
from dcv_benchmark.core.runner import ExperimentRunner
from dcv_benchmark.models.experiments_config import (
    DefenseConfig,
    DetectorConfig,
    ExperimentConfig,
    GenerationStageConfig,
    TargetConfig,
)


@pytest.fixture
def mock_dataset_loader():
    with patch("dcv_benchmark.core.factories.DatasetLoader") as mock_loader:
        mock_ds = MagicMock()
        mock_ds.samples = [MagicMock(id=f"s{i}") for i in range(5)]
        mock_ds.meta.attack_info.payload = (
            f"some payload with {BASELINE_TARGET_KEYWORD}"
        )
        mock_ds.meta.model_dump.return_value = {
            "name": "mock_dataset",
            "version": "1.0",
            "description": "Mocked Dataset",
            "attack_info": {"strategy": "none", "rate": 0.0, "payload": "none"},
        }
        mock_loader.return_value.samples = [mock_ds]
        mock_loader.return_value.load.return_value = mock_ds
        yield mock_loader


@pytest.fixture
def valid_config():
    return ExperimentConfig(
        name="unit_test_exp",
        description="unit test",
        dataset="dummy_dataset",
        target=TargetConfig(
            name="basic_rag",
            defense=DefenseConfig(
                type="deconvolute",
                generation=GenerationStageConfig(
                    canary_detector=DetectorConfig(enabled=True, settings={})
                ),
            ),
            # Minimal other fields to pass validation
            system_prompt={"file": "s", "key": "k"},
            prompt_template={"file": "p", "key": "k"},
        ),
    )


def test_init_does_not_create_dir(tmp_path):
    output_dir = tmp_path / "custom_results"
    _ = ExperimentRunner(output_dir=output_dir)
    assert not output_dir.exists()


def test_run_missing_dataset_path(valid_config, tmp_path):
    runner = ExperimentRunner(output_dir=tmp_path)
    # Ensure BUILT_DATASETS_DIR doesn't incidentally match anything
    with patch("dcv_benchmark.core.factories.BUILT_DATASETS_DIR", tmp_path / "built"):
        valid_config.dataset = "non_existent_dataset"

        with pytest.raises(FileNotFoundError):
            runner.run(valid_config)


def test_run_with_limit(mock_dataset_loader, valid_config, tmp_path):
    """Verify processing stops after limit is reached."""
    runner = ExperimentRunner(output_dir=tmp_path)

    with (
        patch("dcv_benchmark.core.factories.BasicRAG") as MockRAG,
        patch(
            "dcv_benchmark.core.runner.create_experiment_evaluators"
        ) as MockCreateEvaluators,
        patch("dcv_benchmark.core.runner.ReportGenerator"),
    ):
        # Allow creating target
        MockRAG.return_value.invoke.return_value = MagicMock(
            attack_detected=False, used_context=[], content="ok"
        )
        # Mock Evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = MagicMock(passed=True)
        MockCreateEvaluators.return_value = {"mock_eval": mock_evaluator}

        # Dataset has 5 samples (from fixture)
        # Set limit to 2
        runner.run(valid_config, limit=2)

        # Verify BasicRAG invoke called exactly 2 times
        assert MockRAG.return_value.invoke.call_count == 2


def test_run_handles_exception_single_sample(
    mock_dataset_loader, valid_config, tmp_path
):
    """Experiment should continue even if one sample crashes."""
    runner = ExperimentRunner(output_dir=tmp_path)

    with (
        patch("dcv_benchmark.core.factories.BasicRAG") as MockRAG,
        patch(
            "dcv_benchmark.core.runner.create_experiment_evaluators"
        ) as MockCreateEvaluators,
        patch("dcv_benchmark.core.runner.ReportGenerator"),
    ):
        # Mock Evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = MagicMock(passed=True)
        MockCreateEvaluators.return_value = {"mock_eval": mock_evaluator}
        instance = MockRAG.return_value
        # Make BasicRAG raise error on first call, succeed on second
        instance.invoke.side_effect = [
            Exception("Crash"),
            MagicMock(attack_detected=False, used_context=[], content="ok"),
        ]

        runner.run(valid_config, limit=2)

        # Should have attempted both (or up to limit if we didn't crash entirely)
        assert instance.invoke.call_count == 2
        # Verify report generated implies run finished
        # Note: ReportGenerator might rely on reading traces,
        # which we mocking here partially.
        # But run method calls it at the end.
