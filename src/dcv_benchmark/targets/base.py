from abc import ABC, abstractmethod

from dcv_benchmark.models.experiments_config import TargetConfig
from dcv_benchmark.models.responses import TargetResponse


class BaseTarget(ABC):
    """
    Abstract interface for the System Under Test.
    The configuration (system prompt, template, models) is fixed at initialization.
    """

    def __init__(self, config: TargetConfig):
        self.config = config

    @abstractmethod
    def ingest(self, documents: list[str]) -> None:
        """
        Ingests a list of raw text documents into the target's knowledge base.

        This is called once during the setup phase of the experiment.

        The target should:
        1. Split the documents (if needed).
        2. Generate embeddings using the configured model.
        3. Index them in the configured vector store.

        Args:
            documents: A list of strings (the dataset content).
        """
        pass

    @abstractmethod
    def invoke(
        self,
        user_query: str,
        system_prompt: str | None = None,
        forced_context: list[str] | None = None,
        retrieve_only: bool = False,
    ) -> TargetResponse:
        """
        Executes the pipeline for a specific input.

        Args:
            user_query: The query from the user.
            system_prompt: ptional override for the system instruction.
            forced_context: If provided, injects this context (skipping retrieval).
                            Used for testing Generator robustness in isolation.
            retrieve_only: If True, stops after retrieval and returns documents.
                           Used for testing Retriever robustness in isolation.

        Returns:
            A TargetResponse object.
        """
        pass
