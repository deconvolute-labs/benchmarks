from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.data_factory.retrieval import EphemeralRetriever


@pytest.fixture
def mock_chroma_client():
    with patch("dcv_benchmark.data_factory.retrieval.chromadb.EphemeralClient") as mock:
        yield mock


@pytest.fixture
def mock_embedding_fn():
    with patch("dcv_benchmark.data_factory.retrieval.get_embedding_function") as mock:
        # The retriever calls this to get an embedding function instance
        mock.return_value = MagicMock()
        yield mock


def test_ephemeral_retriever_lifecycle(mock_chroma_client, mock_embedding_fn):
    # Setup mocks
    mock_client_instance = mock_chroma_client.return_value
    mock_collection = mock_client_instance.create_collection.return_value

    # Configure query return payload structure:
    # { 'ids': [...], 'embeddings': None, 'documents': [['doc1']], ... }
    # query() returns a dict with 'documents': [ [list of strings] ]
    mock_collection.query.return_value = {"documents": [["The apple is red."]]}

    # Initialize
    retriever = EphemeralRetriever(provider="huggingface", model_name="dummy")

    # Assertions on init
    mock_embedding_fn.assert_called_with("huggingface", "dummy")
    mock_client_instance.create_collection.assert_called()

    docs = ["The apple is red.", "The sky is blue."]
    ids = ["1", "2"]

    # 1. Test Indexing
    retriever.index(docs, ids)
    mock_collection.add.assert_called_once_with(documents=docs, ids=ids)

    # 2. Test Querying
    results = retriever.query("red fruit", k=1)

    # Check that it called the collection query correctly
    mock_collection.query.assert_called_once_with(
        query_texts=["red fruit"], n_results=1
    )
    # Check that it parsed the return value correctly
    assert len(results) == 1
    assert results[0] == "The apple is red."


def test_retriever_empty_query(mock_chroma_client, mock_embedding_fn):
    # Setup mock to return empty
    mock_collection = mock_chroma_client.return_value.create_collection.return_value
    mock_collection.query.return_value = {"documents": [[]]}  # inner list empty

    retriever = EphemeralRetriever()
    results = retriever.query("hello")
    assert results == []


def test_retriever_mismatched_ids(mock_chroma_client, mock_embedding_fn):
    retriever = EphemeralRetriever()
    docs = ["A", "B"]
    ids = ["1"]  # Too few IDs

    with pytest.raises(ValueError, match="Length of IDs"):
        retriever.index(docs, ids)
