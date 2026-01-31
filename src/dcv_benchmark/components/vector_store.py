import uuid
from abc import ABC, abstractmethod
from typing import Any, cast

import chromadb
import openai  # Using it here directly is okay for now
from chromadb.config import Settings

from dcv_benchmark.models.experiments_config import EmbeddingConfig, RetrieverConfig
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """
    Abstract interface for vector database operations.
    """

    @abstractmethod
    def add_documents(self, documents: list[str]) -> None:
        """
        Embeds and indexes a list of text documents.

        Args:
            documents: A list of raw text strings to be added to the store.
        """
        pass

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """
        Retrieves the most relevant documents for a given query.

        Args:
            query: The search query string.

        Returns:
            A list of the top-k matching document strings.
        """
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    Concrete implementation using ChromaDB (Ephemeral/In-Memory).
    """

    def __init__(self, ret_config: RetrieverConfig, emb_config: EmbeddingConfig):
        """
        Initializes the ChromaDB client and embedding provider.

        Args:
            ret_config: Configuration for retrieval (e.g. top_k).
            emb_config: Configuration for the embedding model (provider, model name).
        """
        self.top_k = ret_config.k
        self.model = emb_config.model
        self.provider = emb_config.provider

        # Initialize client
        self.client = chromadb.Client(Settings(is_persistent=False))
        self.collection = self.client.create_collection(name="benchmark_kb")

        if self.provider == "openai":
            self.openai_client = openai.Client()

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            # Sanitize inputs
            clean_texts = [t.replace("\n", " ") for t in texts]
            resp = self.openai_client.embeddings.create(
                input=clean_texts, model=self.model
            )
            return [d.embedding for d in resp.data]
        else:
            # Mock embeddings (1536 dims)
            return [[0.1] * 1536 for _ in texts]

    def add_documents(self, documents: list[str]) -> None:
        """
        Generates embeddings for the documents and upserts them into the collection.
        Uses UUIDs for document IDs.

        Args:
            documents: List of text chunks to index.
        """
        if not documents:
            return

        ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = self._get_embeddings(documents)

        self.collection.add(
            documents=documents, embeddings=cast(Any, embeddings), ids=ids
        )

    def search(self, query: str) -> list[str]:
        """
        Embeds the query and performs a similarity search in ChromaDB.

        Args:
            query: The user query.

        Returns:
            A list of strings representing the top matching context chunks.
        """
        query_embed = self._get_embeddings([query])
        results = self.collection.query(
            query_embeddings=cast(Any, query_embed), n_results=self.top_k
        )
        if results and results["documents"]:
            return results["documents"][0]
        return []


def create_vector_store(
    ret_config: RetrieverConfig | None, emb_config: EmbeddingConfig | None
) -> BaseVectorStore | None:
    """
    Factory function to create a vector store if both retriever and
    embedding configs are present.

    Args:
        ret_config: Retriever configuration object.
        emb_config: Embedding configuration object.

    Returns:
        An initialized BaseVectorStore instance, or None if configuration is missing.
    """

    if not ret_config or not emb_config:
        return None

    if ret_config.provider == "chromadb":
        return ChromaVectorStore(ret_config, emb_config)
    elif ret_config.provider == "mock":
        return None
    return None
