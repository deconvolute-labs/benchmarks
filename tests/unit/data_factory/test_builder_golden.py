from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.data_factory.squad.squad_builder import SquadBuilder
from dcv_benchmark.models.data_factory import DataFactoryConfig, RawSample


@pytest.fixture
def mock_config():
    return DataFactoryConfig(
        dataset_name="repro_ds",
        description="repro",
        source_file="dummy.json",
        attack_strategy="naive",
        attack_payload="dummy",
        attack_rate=0.0,  # Benign to focus on retrieval
        retrieval_k=3,
        truncate_overflow=False,
    )


@pytest.fixture
def mock_loader():
    loader = MagicMock()
    loader.load.return_value = [
        RawSample(
            id="1", query="Q1", source_document="GOLD_CONTENT", reference_answer="Ans1"
        ),
    ]
    return loader


@pytest.fixture
def mock_injector():
    return MagicMock()


@pytest.fixture
def mock_retriever_class():
    with patch(
        "dcv_benchmark.data_factory.squad.squad_builder.EphemeralRetriever"
    ) as MockClass:
        yield MockClass


def test_gold_in_retrieved_k_plus_one(
    mock_config, mock_loader, mock_injector, mock_retriever_class
):
    """
    Test scenario where we retrieve k+1 (4) items.
    The Gold sample is the 4th item (index 3).
    We expect the builder to keep the Gold sample and the top 2 others.
    Total context size should be 3.
    """
    mock_instance = mock_retriever_class.return_value
    # Return 4 items: Top1, Top2, Top3, GOLD_CONTENT
    mock_instance.query.return_value = ["D1", "D2", "D3", "GOLD_CONTENT"]

    # Update config to ensure we are testing k=3 behavior
    mock_config.retrieval_k = 3

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    sample = dataset.samples[0]
    contents = [c.content for c in sample.context]

    mock_instance.query.assert_called_with(query_text="Q1", k=4)

    # 2. Verify Gold is present
    assert "GOLD_CONTENT" in contents

    # 3. Verify size is 3
    assert len(contents) == 3

    # 4. Verify we kept D1, D2 and GOLD (discarded D3 effectively, or whatever
    # priority logic). If we strictly take top k-1 + gold, it should be D1, D2, GOLD.
    # Note: The order in 'contents' might vary depending on implementation
    # (insert at 0 vs preserve order).
    assert set(contents) == {"D1", "D2", "GOLD_CONTENT"}


def test_gold_missing_from_retrieved(
    mock_config, mock_loader, mock_injector, mock_retriever_class
):
    """
    Test scenario where Gold is NOT in the retrieved 4 items.
    We expect replacement of the last item with Gold.
    """
    mock_instance = mock_retriever_class.return_value
    # Return 4 distractors
    mock_instance.query.return_value = ["D1", "D2", "D3", "D4"]
    mock_config.retrieval_k = 3

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    sample = dataset.samples[0]
    contents = [c.content for c in sample.context]

    # Verify k=4 call
    mock_instance.query.assert_called_with(query_text="Q1", k=4)

    # Verify Gold injection
    assert "GOLD_CONTENT" in contents
    assert len(contents) == 3

    # Expect D1, D2, GOLD (D3, D4 dropped)
    assert set(contents) == {"D1", "D2", "GOLD_CONTENT"}


def test_gold_is_top_1(mock_config, mock_loader, mock_injector, mock_retriever_class):
    """
    Test scenario where Gold is the #1 retrieved item.
    Expect it to be kept, and we fill the rest with D1, D2.
    """
    mock_instance = mock_retriever_class.return_value
    # Gold is first
    mock_instance.query.return_value = ["GOLD_CONTENT", "D1", "D2", "D3"]
    mock_config.retrieval_k = 3

    builder = SquadBuilder(mock_loader, mock_injector, mock_config)
    dataset = builder.build()

    sample = dataset.samples[0]
    contents = [c.content for c in sample.context]

    mock_instance.query.assert_called_with(query_text="Q1", k=4)

    assert "GOLD_CONTENT" in contents
    assert len(contents) == 3
    assert set(contents) == {"GOLD_CONTENT", "D1", "D2"}
