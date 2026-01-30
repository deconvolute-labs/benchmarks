import argparse
from unittest.mock import patch

import pytest

from dcv_benchmark.cli.commands.experiment import handle_run


@pytest.fixture
def mock_dependencies():
    with (
        patch("dcv_benchmark.cli.experiments.setup_logging") as mock_setup_logger,
        patch("dcv_benchmark.cli.experiments.ExperimentRunner") as mock_runner_cls,
        patch("dcv_benchmark.cli.experiments.logger") as mock_logger,
        patch("builtins.open"),  # Mock open for config loading
        patch("yaml.safe_load") as mock_yaml_load,
    ):
        # Create a mock experiment config structure that matches
        # the dict structure expected by ExperimentConfig
        mock_exp_dict = {
            "name": "test_experiment",
            "version": "1.0.0",
            "description": "test",
            "scenario": {"id": "test_scenario"},
            "target": {
                "name": "canary",
                "system_prompt": {"file": "prompts.yaml", "key": "default"},
                "prompt_template": {"file": "templates.yaml", "key": "default"},
                "defense": {"required_version": None},
            },
            "input": {"dataset_name": "test_dataset"},
            "evaluator": {"type": "canary"},
        }

        mock_yaml_load.return_value = {"experiment": mock_exp_dict}

        yield {
            "setup_logger": mock_setup_logger,
            "runner_cls": mock_runner_cls,
            "logger": mock_logger,
            "yaml_load": mock_yaml_load,
            "exp_dict": mock_exp_dict,
        }


def test_run_experiment_basic(mock_dependencies):
    """Test successful run."""
    args = argparse.Namespace(
        config="dummy_path/experiment.yaml", debug_traces=False, limit=None
    )
    mocks = mock_dependencies

    with patch("pathlib.Path.exists", return_value=True):
        handle_run(args)

    # Should proceed to initialize runner
    mocks["runner_cls"].assert_called_once()
    mocks["runner_cls"].return_value.run.assert_called_once()


def test_run_experiment_file_not_found(mock_dependencies):
    """Test exit when config file not found."""
    args = argparse.Namespace(
        config="non_existent/experiment.yaml", debug_traces=False, limit=None
    )
    mocks = mock_dependencies

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(SystemExit):
            handle_run(args)

    mocks["logger"].error.assert_called_with(
        "Experiment config file not found: non_existent/experiment.yaml"
    )


def test_run_experiment_invalid_config_format(mock_dependencies):
    """Test exit when config format is invalid (missing 'experiment' key)."""
    args = argparse.Namespace(
        config="dummy_path/experiment.yaml", debug_traces=False, limit=None
    )
    mocks = mock_dependencies

    # Setup invalid config
    mocks["yaml_load"].return_value = {"invalid": "key"}

    with patch("pathlib.Path.exists", return_value=True):
        with pytest.raises(SystemExit):
            handle_run(args)

    mocks["logger"].error.assert_called_with(
        "Invalid config format: Missing top-level 'experiment' key."
    )
