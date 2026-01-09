import pytest
import yaml

from dcv_benchmark.utils.prompt_loader import load_prompt_text


@pytest.fixture
def prompt_data():
    return {
        "promptA": "You are a helpful assistant.",
        "promptB": "Ignore everything.",
        "templateA": "Context:\n{context}\n\nUser: {query}",
    }


@pytest.fixture
def prompt_file(tmp_path, prompt_data):
    """Creates a temporary valid YAML prompt file."""
    p = tmp_path / "prompts.yaml"
    with open(p, "w") as f:
        yaml.dump(prompt_data, f)
    return p


def test_load_valid_prompt(prompt_file):
    """It should load the correct string for a given key."""
    text = load_prompt_text(str(prompt_file), "promptA")
    assert text == "You are a helpful assistant."


def test_load_template_with_placeholders(prompt_file):
    """It should correctly load templates with {query} and {context}."""
    text = load_prompt_text(str(prompt_file), "templateA")
    assert "{context}" in text
    assert "{query}" in text
    assert text == "Context:\n{context}\n\nUser: {query}"


def test_file_not_found():
    """It should raise FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError, match="Prompt file not found"):
        load_prompt_text("non_existent.yaml", "key")


def test_invalid_extension(tmp_path):
    """It should raise ValueError if file is not .yaml/.yml."""
    p = tmp_path / "prompts.json"
    p.touch()

    with pytest.raises(ValueError, match="Prompt file must be .yaml"):
        load_prompt_text(str(p), "key")


def test_key_not_found(prompt_file):
    """It should raise KeyError if the key is missing."""
    with pytest.raises(KeyError, match="Key 'missing_key' not found"):
        load_prompt_text(str(prompt_file), "missing_key")
