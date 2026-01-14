import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.cli.run import run_experiment_command


@pytest.fixture
def mock_dependencies():
    with (
        patch("dcv_benchmark.cli.run.setup_logger") as mock_setup_logger,
        patch("dcv_benchmark.cli.run.resolve_scenario_paths") as mock_resolve_path,
        patch("dcv_benchmark.cli.run.load_experiment") as mock_load_exp,
        patch("dcv_benchmark.cli.run.print_experiment_header") as mock_print_header,
        patch("dcv_benchmark.cli.run.ensure_dataset_exists") as mock_ensure_data,
        patch("dcv_benchmark.cli.run.ExperimentRunner") as mock_runner_cls,
        patch("dcv_benchmark.cli.run.logger") as mock_logger,
        patch("dcv_benchmark.cli.run.version") as mock_version,
    ):
        # Setup common mocks
        mock_resolve_path.return_value = Path("dummy_path/experiment.yaml")

        # Create a mock experiment config structure
        # Use a plain Mock or recursive set for complex nested structures
        # when not strictly testing the model itself
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {}
        mock_config.name = "test_experiment"

        # Ensure deep structure exists for defense version check
        mock_config.target.defense.required_version = None

        mock_load_exp.return_value = mock_config
        mock_version.return_value = "1.0.0"

        yield {
            "setup_logger": mock_setup_logger,
            "resolve_path": mock_resolve_path,
            "load_exp": mock_load_exp,
            "print_header": mock_print_header,
            "ensure_data": mock_ensure_data,
            "runner_cls": mock_runner_cls,
            "logger": mock_logger,
            "version": mock_version,
            "config": mock_config,
        }


def test_run_experiment_version_match(mock_dependencies):
    """Test successful run when version matches required version."""
    args = argparse.Namespace(
        target="canary", debug=False, dry_run=False, limit=None, output_dir=None
    )
    mocks = mock_dependencies

    # Setup required version to match installed version
    mocks["config"].target.defense.required_version = "1.0.0"
    mocks["version"].return_value = "1.0.0"

    run_experiment_command(args)

    # Should proceed to initialize runner
    mocks["runner_cls"].assert_called_once()
    mocks["runner_cls"].return_value.run.assert_called_once()


def test_run_experiment_version_mismatch(mock_dependencies):
    """Test ImportError when version does not match required version."""
    args = argparse.Namespace(
        target="canary", debug=False, dry_run=False, limit=None, output_dir=None
    )
    mocks = mock_dependencies

    # Setup mismatch
    mocks["config"].target.defense.required_version = "2.0.0"
    mocks["version"].return_value = "1.0.0"

    # The code logs a warning but proceeds with execution.
    run_experiment_command(args)

    # Verify warning logging
    mocks["logger"].warning.assert_called()
    call_args = mocks["logger"].warning.call_args[0][0]
    assert "Deconvolute version mismatch" in call_args

    # Verify that it proceeded to run the experiment
    mocks["runner_cls"].assert_called_once()
    mocks["runner_cls"].return_value.run.assert_called_once()


def test_run_experiment_no_required_version(mock_dependencies):
    """Test successful run when no version is required."""
    args = argparse.Namespace(
        target="canary", debug=False, dry_run=False, limit=None, output_dir=None
    )
    mocks = mock_dependencies

    # Setup no required version
    mocks["config"].target.defense.required_version = None

    run_experiment_command(args)

    # Should not call version() if not required (optional check, but good for
    # performance)
    mocks["version"].assert_not_called()

    # Should proceed
    mocks["runner_cls"].assert_called_once()
