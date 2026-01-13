import json

import pytest

from dcv_benchmark.data_factory.loaders import SquadLoader
from dcv_benchmark.models.data_factory import RawSample


def test_squad_loader_valid(tmp_path):
    # Create a dummy SQuAD-like JSON file
    data = {
        "data": [
            {
                "id": "101",
                "query": "What is AI?",
                "source_document": "AI is artificial intelligence.",
                "reference_answer": "Artificial Intelligence",
                "title": "AI Stuff",
            }
        ]
    }
    f = tmp_path / "squad.json"
    f.write_text(json.dumps(data), encoding="utf-8")

    loader = SquadLoader()
    samples = loader.load(str(f))

    assert len(samples) == 1
    s = samples[0]
    assert isinstance(s, RawSample)
    assert s.id == "101"
    assert s.query == "What is AI?"
    assert s.metadata["title"] == "AI Stuff"


def test_squad_loader_malformed(tmp_path):
    # Missing 'source_document'
    data = {
        "data": [
            {"id": "1", "query": "Bad sample"},
            {"id": "2", "query": "Good sample", "source_document": "Content"},
        ]
    }
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(data), encoding="utf-8")

    loader = SquadLoader()
    samples = loader.load(str(f))

    # Should skip the bad one, keep the good one
    assert len(samples) == 1
    assert samples[0].id == "2"


def test_squad_loader_file_not_found():
    loader = SquadLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("non_existent_file.json")
