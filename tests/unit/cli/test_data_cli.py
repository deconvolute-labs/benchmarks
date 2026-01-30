import argparse
from unittest.mock import patch

import pytest

from dcv_benchmark.cli.commands.data import handle_build, handle_download


@pytest.fixture
def mock_data_dependencies():
    with (
        patch("dcv_benchmark.cli.data.download_squad") as mock_dl_squad,
        patch("dcv_benchmark.cli.data.download_bipia") as mock_dl_bipia,
        patch("dcv_benchmark.cli.data.DatasetBuilder") as mock_builder_cls,
        patch("dcv_benchmark.cli.data.AttackInjector") as mock_injector,
        patch("dcv_benchmark.cli.data.SquadLoader") as mock_loader,
        patch("dcv_benchmark.cli.data.logger") as mock_logger,
        patch("builtins.open"),
        patch("yaml.safe_load") as mock_yaml_load,
    ):
        yield {
            "dl_squad": mock_dl_squad,
            "dl_bipia": mock_dl_bipia,
            "builder_cls": mock_builder_cls,
            "injector": mock_injector,
            "loader": mock_loader,
            "logger": mock_logger,
            "yaml_load": mock_yaml_load,
        }


def test_handle_download_squad(mock_data_dependencies):
    """Test downloading squad dataset."""
    args = argparse.Namespace(source="squad", output_dir=None)
    mocks = mock_data_dependencies

    handle_download(args)

    mocks["dl_squad"].assert_called_once()
    mocks["dl_bipia"].assert_not_called()
    mocks["logger"].info.assert_called()


def test_handle_download_bipia(mock_data_dependencies):
    """Test downloading bipia dataset."""
    args = argparse.Namespace(source="bipia", output_dir="custom/path")
    mocks = mock_data_dependencies

    handle_download(args)

    mocks["dl_bipia"].assert_called_once()
    mocks["dl_squad"].assert_not_called()
    # verify output dir was passed (logic is inside function, but we can check
    # calls if we mocked Path, but primarily we just want to ensure the right
    # downloader is called)


def test_handle_download_unknown(mock_data_dependencies):
    """Test unknown source exits."""
    args = argparse.Namespace(source="unknown", output_dir=None)
    mocks = mock_data_dependencies

    with pytest.raises(SystemExit):
        handle_download(args)

    mocks["logger"].error.assert_called_with(
        "Unknown source: 'unknown'. Options: squad, bipia"
    )


def test_handle_build_success(mock_data_dependencies):
    """Test successful build flow."""
    args = argparse.Namespace(config="config.yaml", name=None, overwrite=False)
    mocks = mock_data_dependencies

    # Setup mocks
    mocks["yaml_load"].return_value = {
        "dataset_name": "test_ds",
        "description": "Test description",
        "source_file": "corpus.json",
        "attack_strategy": "none",
        "attack_rate": 0.0,
        "attack_payload": "test_payload",
    }

    # Mock Path exists behaviors
    with (
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.mkdir"),
        # We need to mock unlink for cleanup if it were called (it shouldn't be here)
        patch("pathlib.Path.unlink"),
    ):
        # 1. Config exists -> True
        # 2. Output file exists -> False
        mock_exists.side_effect = [True, False]

        handle_build(args)

    mocks["builder_cls"].assert_called_once()
    mocks["builder_cls"].return_value.build.assert_called_once()
    mocks["builder_cls"].return_value.save.assert_called_once()

    # Ensure config was NOT copied
    # mock_shutil is no longer available, nor used.
    pass


def test_handle_build_overwrite_denied(mock_data_dependencies):
    """Test build failure when target exists and no overwrite flag."""
    args = argparse.Namespace(config="config.yaml", name="test_ds", overwrite=False)
    mocks = mock_data_dependencies
    mocks["yaml_load"].return_value = {
        "dataset_name": "test_ds",
        "description": "Test description",
        "source_file": "corpus.json",
        "attack_strategy": "none",
        "attack_payload": "test_payload",
    }

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.mkdir"),
    ):
        with pytest.raises(SystemExit):
            handle_build(args)

    mocks["logger"].error.assert_called()
    # Check for updated error message
    assert "Dataset artifact" in mocks["logger"].error.call_args[0][0]
    assert "already exists" in mocks["logger"].error.call_args[0][0]


def test_handle_build_overwrite_allowed(mock_data_dependencies):
    """Test build overwrites when flag is set."""
    args = argparse.Namespace(config="config.yaml", name="test_ds", overwrite=True)
    mocks = mock_data_dependencies
    mocks["yaml_load"].return_value = {
        "dataset_name": "test_ds",
        "description": "Test description",
        "source_file": "corpus.json",
        "attack_strategy": "none",
        "attack_payload": "test_payload",
    }

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.unlink") as mock_unlink,
    ):
        handle_build(args)

    # Should have called unlink (not rmtree anymore)
    mock_unlink.assert_called()

    # Should proceed to build
    mocks["builder_cls"].assert_called_once()
