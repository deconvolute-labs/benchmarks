import uuid
from typing import Any, Literal

import chromadb
from chromadb import EmbeddingFunction

from dcv_benchmark.components.embedder import get_embedding_function
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


class EphemeralRetriever:
    """
    A temporary, in-memory vector store used solely for generating datasets.

    It indexes the provided corpus on-the-fly and retrieves relevant chunks
    to serve as 'distractors' in the final RAG dataset.
    """

    def __init__(
        self,
        provider: Literal["huggingface", "openai"] = "huggingface",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "dataset_generation",
    ):
        """
        Initializes the ephemeral retriever with a specific embedding model.

        Args:
            provider: The embedding provider ('huggingface' or 'openai').
            model_name: The specific model ID to use.
            collection_name: Name of the temporary ChromaDB collection.
        """
        # EphemeralClient is in-memory only; data vanishes when script ends.
        # Note: This tool relies on ChromaDB for convenience in the build step.
        self.client = chromadb.EphemeralClient()

        # Factory to get the correct embedding function
        self.embedding_fn: EmbeddingFunction[Any] = get_embedding_function(
            provider, model_name
        )

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, documents: list[str], ids: list[str] | None = None) -> None:
        """
        Embeds and indexes the provided documents into the in-memory vector store.

        This method handles batching automatically (via ChromaDB) and ensures
        that all documents are available for subsequent similarity queries.

        Args:
            documents: A list of text strings (chunks or paragraphs) to be indexed.
            ids: An optional list of unique string IDs corresponding to the documents.
                 If not provided, UUIDs will be generated automatically.

        Raises:
            ValueError: If the length of `ids` does not match the length of `documents`.
        """
        count = len(documents)
        if count == 0:
            logger.warning("No documents provided to index.")
            return

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(count)]

        if len(ids) != count:
            raise ValueError("Length of IDs must match length of documents.")

        logger.info(f"Indexing {count} documents into ephemeral store...")

        batch_size = 100
        for i in range(0, count, batch_size):
            end = min(i + batch_size, count)
            self.collection.add(documents=documents[i:end], ids=ids[i:end])

        logger.info("Indexing complete.")

    def query(self, query_text: str, k: int = 3) -> list[str]:
        """
        Retrieves the top-k most semantically similar documents for the given query.

        Args:
            query_text: The input query string to search for.
            k: The number of nearest neighbors (documents) to retrieve.
               Defaults to 3.

        Returns:
            A list of document strings sorted by relevance (descending).
            Returns an empty list if the collection is empty or no matches are found.
        """
        results = self.collection.query(query_texts=[query_text], n_results=k)

        if results["documents"] and results["documents"][0]:
            return results["documents"][0]

        return []
