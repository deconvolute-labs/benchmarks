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
        Executes the target pipeline for a single interaction.

        This method encapsulates the entire RAG flow: Retrieval (if enabled),
        Input Defense (e.g. Canary injection), LLM Generation, and Output Defense.

        Args:
            user_query (str): The final user input string (e.g. a question or command).
            system_prompt (str | None, optional): An override for the system instruction.
                If None, the target uses its configured default system prompt.
            forced_context (list[str] | None, optional): A list of context strings to
                inject directly into the prompt, bypassing the retrieval step.
                Used to test Generator robustness in isolation or to simulate
                specific retrieval outcomes (e.g. "Oracle" tests).
            retrieve_only (bool, optional): If True, the pipeline stops after the
                retrieval step. The returned TargetResponse will contain the
                retrieved chunks but an empty generation. Defaults to False.

        Returns:
            TargetResponse: A unified object containing the model's output text,
            metadata, and any security signals (e.g. `attack_detected=True`).
        """
        pass
