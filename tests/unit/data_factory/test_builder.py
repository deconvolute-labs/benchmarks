from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.data_factory.squad.squad_builder import SquadBuilder
from dcv_benchmark.models.data_factory import DataFactoryConfig, RawSample
from dcv_benchmark.models.dataset import Dataset


@pytest.fixture
def mock_config():
    return DataFactoryConfig(
        dataset_name="test_ds",
        description="test",
        source_file="dummy.json",
        attack_strategy="naive",
        attack_rate=0.5,
        attack_payload="PAYLOAD",
        retrieval_k=3,
        truncate_overflow=False,
        flooding_repetitions=5,
    )


@pytest.fixture
def mock_loader():
    loader = MagicMock()
    # Return 2 samples
    loader.load.return_value = [
        RawSample(id="1", query="Q1", source_document="Gold1", reference_answer="Ans1"),
        RawSample(id="2", query="Q2", source_document="Gold2", reference_answer="Ans2"),
    ]
    return loader


@pytest.fixture
def mock_injector():
    injector = MagicMock()
    injector.inject.return_value = "POISONED_CONTENT"
    return injector


@pytest.fixture
def mock_retriever_class():
    with patch(
        "dcv_benchmark.data_factory.squad.squad_builder.EphemeralRetriever"
    ) as MockClass:
        yield MockClass


def test_build_workflow(mock_config, mock_loader, mock_injector, mock_retriever_class):
    """Test the complete build loop with mocked components."""
    # Setup Retriever Mock Instance
    mock_retriever_instance = mock_retriever_class.return_value
    # Simulate finding distractors
    mock_retriever_instance.query.return_value = [
        "Distractor A",
        "Distractor B",
        "Distractor C",
    ]

    # Force Attack Rate to 1.0 to ensure injection happens
    mock_config.attack_rate = 1.0

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    # Verify Loader Interaction
    mock_loader.load.assert_called_once_with("dummy.json")

    # Verify Metadata Population
    assert dataset.meta.attack_info is not None
    assert dataset.meta.attack_info.strategy == "naive"
    assert dataset.meta.attack_info.payload == "PAYLOAD"
    # Verify the rate reflects the config passed to builder (1.0 override)
    assert dataset.meta.attack_info.rate == 1.0
    assert dataset.meta.attack_info.configuration["truncate"] is False

    # Verify Indexing
    # Should index ["Gold1", "Gold2"]
    mock_retriever_instance.index.assert_called_once()
    indexed_docs = mock_retriever_instance.index.call_args[1]["documents"]
    assert set(indexed_docs) == {"Gold1", "Gold2"}

    # Verify Result Structure
    assert isinstance(dataset, Dataset)
    assert len(dataset.samples) == 2

    sample = dataset.samples[0]
    assert sample.id == "1"
    assert sample.query == "Q1"
    assert sample.sample_type == "attack"

    # Verify Injection
    # Since rate is 1.0, one chunk should be poisoned
    mock_injector.inject.assert_called()
    assert any(c.is_malicious for c in sample.context)
    assert any(c.content == "POISONED_CONTENT" for c in sample.context)


def test_gold_chunk_always_present(
    mock_config, mock_loader, mock_injector, mock_retriever_class
):
    """Test that the source_document (Gold) is forced into the context if missing."""
    mock_retriever_instance = mock_retriever_class.return_value
    # Retriever returns ONLY distractors, NO Gold chunk
    mock_retriever_instance.query.return_value = [
        "Distractor A",
        "Distractor B",
        "Distractor C",
    ]

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    sample = dataset.samples[0]
    # The gold chunk for sample 1 is "Gold1"
    # It must be present in the final context list
    contents = [c.content for c in sample.context]
    assert "Gold1" in contents

    # It should have replaced one distractor, keeping count at K=3
    assert len(contents) == 3


def test_attack_rate_zero(
    mock_config, mock_loader, mock_injector, mock_retriever_class
):
    """Test that attack_rate=0.0 produces only benign samples."""
    mock_config.attack_rate = 0.0
    mock_retriever_instance = mock_retriever_class.return_value
    mock_retriever_instance.query.return_value = ["Gold1", "D1", "D2"]

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    for sample in dataset.samples:
        assert sample.sample_type == "benign"
        assert not any(c.is_malicious for c in sample.context)

    # Injector should NEVER be called
    mock_injector.inject.assert_not_called()


def test_save_dataset(
    mock_config, mock_loader, mock_injector, mock_retriever_class, tmp_path
):
    """Test saving the dataset to JSON."""
    mock_instance = mock_retriever_class.return_value
    mock_instance.query.return_value = ["Doc A", "Doc B", "Doc C"]

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    output_path = tmp_path / "dataset.json"
    builder.save(dataset, output_path)

    assert output_path.exists()
    content = output_path.read_text("utf-8")
    assert "test_ds" in content
    # Verify samples were written
    assert "Doc A" in content
