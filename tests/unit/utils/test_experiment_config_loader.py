from pathlib import Path

import pytest
import yaml

from dcv_benchmark.utils.experiment_loader import load_experiment


@pytest.fixture
def valid_experiment_data():
    return {
        "name": "test_exp",
        "description": "test",
        "dataset": "squad_val",
        "target": {
            "name": "toy_rag",
            "system_prompt": {"file": "prompts.yaml", "key": "promptA"},
            "prompt_template": {"file": "templates.yaml", "key": "templateA"},
            "defense": {"type": "deconvolute"},
            "llm": {"provider": "openai", "model": "gpt-4"},
        },
    }


@pytest.fixture
def experiment_file(tmp_path, valid_experiment_data):
    """Creates a temporary valid YAML file."""
    p = tmp_path / "valid.yaml"
    with open(p, "w") as f:
        yaml.dump(valid_experiment_data, f)
    return p


def test_load_valid_config(experiment_file, valid_experiment_data):
    """It should load and return the experiment object."""
    experiment = load_experiment(experiment_file)
    assert experiment.name == "test_exp"
    assert experiment.name == "test_exp"
    # assert experiment.target.defense.type == "deconvolute" # Field removed


def test_file_not_found():
    """It should raise FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        load_experiment(Path("ghost.yaml"))


def test_invalid_yaml_syntax(tmp_path):
    """It should raise ValueError for broken YAML."""
    p = tmp_path / "broken.yaml"
    p.write_text("experiment: [unclosed list", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse YAML"):
        load_experiment(p)


def test_validation_missing_required_section(tmp_path, valid_experiment_data):
    """It should detect missing required sections ( 'target')."""
    # Remove 'target' from the valid data
    del valid_experiment_data["target"]

    p = tmp_path / "incomplete.yaml"
    with open(p, "w") as f:
        yaml.dump(valid_experiment_data, f)

    with pytest.raises(ValueError, match="Invalid experiment configuration"):
        load_experiment(p)
