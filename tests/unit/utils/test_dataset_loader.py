import json

import pytest
from pydantic import ValidationError

from dcv_benchmark.utils.dataset_loader import DatasetLoader


@pytest.fixture
def valid_dataset_json():
    return {
        "meta": {
            "name": "test_dataset",
            "version": "1.0",
            "description": "A test dataset",
            "author": "Test Author",
        },
        "samples": [
            {
                "id": "sample_1",
                "query": "What is the capital of France?",
                "sample_type": "benign",
                "expected_behavior": "answer",
                "context": [
                    {
                        "id": "chunk_1",
                        "content": "Paris is the capital of France.",
                        "is_malicious": False,
                    }
                ],
            }
        ],
    }


@pytest.fixture
def dataset_file(tmp_path, valid_dataset_json):
    """Creates a temporary valid JSON dataset file."""
    p = tmp_path / "dataset.json"
    with open(p, "w") as f:
        json.dump(valid_dataset_json, f)
    return p


def test_load_valid_dataset(dataset_file):
    """It should correctly load and validate a compliant JSON dataset."""
    loader = DatasetLoader(str(dataset_file))
    dataset = loader.load()

    assert dataset.meta.name == "test_dataset"
    assert len(dataset.samples) == 1
    assert dataset.samples[0].id == "sample_1"
    assert dataset.samples[0].context[0].content == "Paris is the capital of France."


def test_file_not_found():
    """It should raise FileNotFoundError for non-existent paths."""
    loader = DatasetLoader("non_existent_file.json")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_invalid_json_syntax(tmp_path):
    """It should raise ValueError for malformed JSON."""
    p = tmp_path / "broken.json"
    p.write_text("{ incomplete json", encoding="utf-8")

    loader = DatasetLoader(str(p))
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.load()


def test_validation_missing_fields(tmp_path, valid_dataset_json):
    """It should raise ValidationError if required fields are missing."""
    # Remove required 'meta' field
    del valid_dataset_json["meta"]

    p = tmp_path / "invalid_schema.json"
    with open(p, "w") as f:
        json.dump(valid_dataset_json, f)

    loader = DatasetLoader(str(p))
    with pytest.raises(ValidationError):
        loader.load()
