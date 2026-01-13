from unittest.mock import patch

import pytest

from dcv_benchmark.components.embedder import get_embedding_function


def test_get_embedding_hf():
    with patch(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    ) as MockHF:
        fn = get_embedding_function("huggingface", "model_name")
        MockHF.assert_called_once_with(model_name="model_name")
        assert fn == MockHF.return_value


def test_get_embedding_openai(monkeypatch):
    # Mock API Key presence
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

    with patch("chromadb.utils.embedding_functions.OpenAIEmbeddingFunction") as MockOAI:
        fn = get_embedding_function("openai", "gpt-embedding")
        MockOAI.assert_called_once_with(
            api_key="sk-fake-key", model_name="gpt-embedding"
        )
        assert fn == MockOAI.return_value


def test_get_embedding_openai_missing_key(monkeypatch):
    # Ensure no API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_embedding_function("openai", "model")


def test_get_embedding_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_function("deepmind", "model")
