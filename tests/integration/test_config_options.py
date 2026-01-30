import datetime
import json
from unittest.mock import MagicMock

import pytest

from dcv_benchmark.core.runner import ExperimentRunner
from dcv_benchmark.models.dataset import (
    AttackInfo,
    BenchmarkSample,
    Dataset,
    DatasetMeta,
)
from dcv_benchmark.models.evaluation import SecurityEvaluationResult
from dcv_benchmark.models.experiments_config import (
    ExperimentConfig,
    InputConfig,
    ScenarioConfig,
    TargetConfig,
)
from dcv_benchmark.models.responses import TargetResponse


@pytest.fixture
def mock_dataset_sample():
    return BenchmarkSample(
        id="test_id",
        sample_type="benign",
        query="test query",
        labels=[],
        context=[{"content": "test context", "source": "test", "id": "1"}],
    )


@pytest.fixture
def mock_target_response():
    return TargetResponse(
        content="test content",
        raw_content="test raw content",
        used_context=["test context"],
    )


def test_default_dataset_path_resolution(tmp_path, monkeypatch):
    """
    Test that the runner falls back to workspace/datasets/built/{name}/dataset.json
    if input.dataset_name is missing.
    """
    # Setup Mock Dataset
    dataset_name = "test_default_ds"
    workspace_dir = tmp_path / "workspace"
    built_ds_dir = workspace_dir / "datasets" / "built" / dataset_name
    built_ds_dir.mkdir(parents=True)

    ds_file = built_ds_dir / "dataset.json"
    ds_content = {
        "meta": {
            "name": dataset_name,
            "version": "1.0",
            "description": "Test DS",
            "author": "tester",
            "attack_info": {"strategy": "none", "rate": 0.0, "payload": "payload"},
        },
        "samples": [],
    }
    with open(ds_file, "w") as f:
        json.dump(ds_content, f)

    # Mock Constants
    monkeypatch.setattr(
        "dcv_benchmark.constants.BUILT_DATASETS_DIR",
        workspace_dir / "datasets" / "built",
    )

    # Create Config without dataset_name
    config = ExperimentConfig(
        name=dataset_name,
        target=TargetConfig(
            name="basic_rag",
            system_prompt={"file": "foo", "key": "bar"},
            prompt_template={"file": "foo", "key": "bar"},
            defense={"type": "deconvolute", "canary": {"enabled": True}},
        ),
        scenario=ScenarioConfig(id="test"),
        evaluator={"type": "canary"},
    )
    # Ensure input.dataset_name is None
    config.input.dataset_name = None

    # Run (dry run with 0 samples effectively)
    runner = ExperimentRunner(output_dir=tmp_path / "results")

    mock_loader_cls = MagicMock()
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.meta.name = "mocked"
    mock_dataset_instance.meta.version = "1"
    mock_dataset_instance.meta.description = "mocked"
    mock_dataset_instance.samples = []

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = mock_dataset_instance
    mock_loader_cls.return_value = mock_loader_instance

    monkeypatch.setattr("dcv_benchmark.core.runner.DatasetLoader", mock_loader_cls)

    try:
        runner.run(config, limit=0)
    except Exception:  # noqa: S110
        pass

    expected_path = str(built_ds_dir / "dataset.json")
    mock_loader_cls.assert_called_with(expected_path)


def test_debug_traces_flag(
    tmp_path, monkeypatch, mock_dataset_sample, mock_target_response
):
    """
    Test that debug_traces=False hides content, and True shows it.
    """
    mock_dataset = Dataset(
        meta=DatasetMeta(
            name="test",
            version="1",
            description="desc",
            author="tester",
            attack_info=AttackInfo(strategy="none", rate=0.0, payload="foo"),
        ),
        samples=[mock_dataset_sample],
    )

    mock_loader_cls = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = mock_dataset
    mock_loader_cls.return_value = mock_loader_instance

    monkeypatch.setattr("dcv_benchmark.core.runner.DatasetLoader", mock_loader_cls)

    mock_target_cls = MagicMock()
    mock_target_instance = MagicMock()

    # We must return a fresh copy every time because runner mutates the
    # response in trace and if we return the same instance, previous test case
    # mutation affects next test case.
    mock_target_instance.invoke.side_effect = (
        lambda *args, **kwargs: mock_target_response.model_copy(deep=True)
    )

    mock_target_cls.return_value = mock_target_instance

    monkeypatch.setattr("dcv_benchmark.core.runner.BasicRAG", mock_target_cls)

    mock_evaluator_cls = MagicMock()
    mock_evaluator_instance = MagicMock()
    mock_evaluator_instance.evaluate.return_value = SecurityEvaluationResult(
        type="security", passed=True, reason="ok", score=1.0, vulnerability_type="none"
    )
    mock_evaluator_cls.return_value = mock_evaluator_instance
    monkeypatch.setattr("dcv_benchmark.core.runner.CanaryEvaluator", mock_evaluator_cls)

    config = ExperimentConfig(
        name="test_exp",
        input=InputConfig(dataset_name="dummy"),
        target=TargetConfig(
            name="basic_rag",
            system_prompt={"file": "foo", "key": "bar"},
            prompt_template={"file": "foo", "key": "bar"},
            defense={"type": "deconvolute", "canary": {"enabled": True}},
        ),
        scenario=ScenarioConfig(id="test"),
        evaluator={"type": "canary"},
    )

    runner = ExperimentRunner(output_dir=tmp_path / "results")

    # Mock datetime to avoid directory collision
    # We need to return different times for each call
    t1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    t2 = datetime.datetime(2023, 1, 1, 12, 1, 0)

    real_datetime = datetime.datetime

    class MockDatetime(real_datetime):
        _calls = 0

        @classmethod
        def now(cls, tz=None):
            cls._calls += 1
            if cls._calls == 1:
                return t1
            return t2

    # Modify the imported module in runner
    monkeypatch.setattr("dcv_benchmark.core.runner.datetime.datetime", MockDatetime)

    # TEST CASE 1: debug_traces = False (Default)
    run_dir_1 = runner.run(config, limit=1, debug_traces=False)
    trace_file_1 = run_dir_1 / "traces.jsonl"

    with open(trace_file_1) as f:
        trace1 = json.loads(f.readline())

    assert trace1["user_query"] is None, (
        "user_query should be None when debug_traces is False"
    )
    assert trace1["response"]["content"] is None, "response content should be None"
    assert trace1["response"]["raw_content"] is None, "raw content should be None"
    assert trace1["response"]["used_context"] == [], "used_context should be empty"

    # TEST CASE 2: debug_traces = True
    run_dir_2 = runner.run(config, limit=1, debug_traces=True)
    trace_file_2 = run_dir_2 / "traces.jsonl"

    with open(trace_file_2) as f:
        trace2 = json.loads(f.readline())

    assert trace2["user_query"] == "test query"
    assert trace2["response"]["content"] == "test content"
    assert trace2["response"]["raw_content"] == "test raw content"
    assert trace2["response"]["used_context"] == ["test context"]
