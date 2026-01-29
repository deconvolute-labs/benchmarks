import argparse
from unittest.mock import patch

import pytest

from dcv_benchmark.cli.data import handle_build, handle_download


@pytest.fixture
def mock_data_dependencies():
    with (
        patch("dcv_benchmark.cli.data.download_squad") as mock_dl_squad,
        patch("dcv_benchmark.cli.data.download_bipia") as mock_dl_bipia,
        patch("dcv_benchmark.cli.data.DatasetBuilder") as mock_builder_cls,
        patch("dcv_benchmark.cli.data.AttackInjector") as mock_injector,
        patch("dcv_benchmark.cli.data.SquadLoader") as mock_loader,
        patch("dcv_benchmark.cli.data.shutil") as mock_shutil,
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
            "shutil": mock_shutil,
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
    # verify output dir was passed (logic is inside function, but we can check calls if we mocked Path,
    # but primarily we just want to ensure the right downloader is called)


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
    ):
        # 1. Config exists
        # 2. Target dir does NOT exist (so no overwrite check needed)
        def exists_side_effect():
            # First check: config path exists -> True
            yield True
            # Second check: target dir exists -> False
            yield False

        mock_exists.side_effect = exists_side_effect()

        handle_build(args)

    mocks["builder_cls"].assert_called_once()
    mocks["builder_cls"].return_value.build.assert_called_once()
    mocks["builder_cls"].return_value.save.assert_called_once()


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
    ):
        with pytest.raises(SystemExit):
            handle_build(args)

    mocks["logger"].error.assert_called()
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
    ):
        handle_build(args)

    # Should have called rmtree
    mocks["shutil"].rmtree.assert_called_once()
    # Should proceed to build
    mocks["builder_cls"].assert_called_once()
