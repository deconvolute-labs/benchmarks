from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.components.vector_store import ChromaVectorStore, create_vector_store
from dcv_benchmark.models.experiments_config import EmbeddingConfig, RetrieverConfig


@pytest.fixture
def chroma_config():
    return RetrieverConfig(provider="chromadb", k=3, chunk_size=500)


@pytest.fixture
def embedding_config():
    return EmbeddingConfig(provider="mock", model="text-embedding-3-small")


def test_create_chroma_store(chroma_config, embedding_config):
    """It should return a ChromaVectorStore when provider is chroma."""
    # Patch chromadb.Client to avoid real initialization
    with (
        patch("dcv_benchmark.components.vector_store.chromadb.Client"),
        patch("dcv_benchmark.components.vector_store.Settings"),
    ):
        store = create_vector_store(chroma_config, embedding_config)

    assert isinstance(store, ChromaVectorStore)
    assert store.top_k == 3


def test_missing_configs_graceful_return():
    """It should return None if configs are missing."""
    chroma_conf = RetrieverConfig(provider="chromadb")
    emb_conf = EmbeddingConfig(provider="mock", model="test")

    # Both missing
    assert create_vector_store(None, None) is None
    # Retriever missing
    assert create_vector_store(None, emb_conf) is None
    # Embedding missing
    assert create_vector_store(chroma_conf, None) is None


def test_mock_provider_returns_none(embedding_config):
    """It should return None (currently) for mock provider."""
    mock_ret_config = RetrieverConfig(provider="mock")

    store = create_vector_store(mock_ret_config, embedding_config)
    assert store is None


def test_unknown_provider_returns_none(embedding_config):
    """It should return None for unknown providers."""
    # Simulate an unknown provider if it passed validation
    unknown_conf = MagicMock()
    unknown_conf.provider = "faiss"

    store = create_vector_store(unknown_conf, embedding_config)
    assert store is None
