import os
from typing import Any, Literal

from chromadb import EmbeddingFunction
from chromadb.utils import embedding_functions

from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_function(
    provider: Literal["huggingface", "openai"], model_name: str
) -> EmbeddingFunction[Any]:
    """
    Factory to return a ChromaDB-compatible embedding function based on the provider.

    Args:
        provider: 'huggingface' (local) or 'openai' (API).
        model_name: The specific model ID.

    Returns:
        A callable EmbeddingFunction object compatible with ChromaDB.

    Raises:
        ValueError: If the provider is unknown or API keys are missing.
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI embeddings."
            )
        logger.info(f"Using OpenAI embeddings: {model_name}")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name=model_name
        )

    elif provider == "huggingface":
        # Uses local sentence-transformers (CPU friendly, good baseline)
        logger.info(f"Using HuggingFace embeddings (local): {model_name}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
