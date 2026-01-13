import logging

from deconvolute import Canary, CanaryResult

from dcv_benchmark.components.llms import BaseLLM, create_llm
from dcv_benchmark.components.vector_store import create_vector_store
from dcv_benchmark.models.experiments_config import TargetConfig
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.targets.base import BaseTarget
from dcv_benchmark.utils.prompt_loader import load_prompt_text

logger = logging.getLogger()


class BasicRAG(BaseTarget):
    """
    A reference RAG implementation for benchmarking.

    This class simulates a standard RAG pipeline with optional modular components:
    - Retriever (ChromaDB + Embeddings)
    - Generator (LLM)
    - Defense Layers (Deconvolute SDK)
    """

    def __init__(self, config: TargetConfig):
        """
        Sets up the pipeline components based on the provided configuration.
        Initializes LLM, Vector Store, and Defense layers if enabled.

        Args:
            config: The full TargetConfig configuration object.
        """
        super().__init__(config)

        # Setup LLM
        self.llm: BaseLLM | None = None
        if config.llm:
            logger.debug(f"Initializing LLM: {config.llm.provider}")
            self.llm = create_llm(config.llm)

        # Setup vector store
        self.vector_store = None
        if config.embedding and config.retriever:
            self.vector_store = create_vector_store(config.retriever, config.embedding)
            logger.debug("Vector Store initialized.")
        else:
            logger.debug("No Retriever configured. Running in Generator-only mode.")

        # Setup Deconvolute defense
        self.canary = Canary()
        self.canary_enabled = False  # For tracking

        for layer in config.defense.layers:
            if layer.type == "canary" and layer.enabled:
                self.canary_enabled = True
                logger.info("Deconvolute Canary defense ENABLED.")

        # Load system prompt
        self.system_prompt: str = load_prompt_text(
            path=config.system_prompt.path,
            key=config.system_prompt.key,
        )

        # Load prompt template
        self.prompt_template: str = load_prompt_text(
            path=config.prompt_template.path,
            key=config.prompt_template.key,
        )

    def ingest(self, documents: list[str]) -> None:
        """
        Populates the vector store with the provided dataset.

        This method is idempotent-ish for the benchmark run (adds to the ephemeral DB).
        If no vector store is configured, this operation logs a warning and skips.

        Args:
            documents: A list of text strings (knowledge base) to index.
        """
        if not self.vector_store:
            logger.warning("Ingest called but no Vector Store is configured. Skipping.")
            return

        logger.info(f"Ingesting {len(documents)} documents...")
        self.vector_store.add_documents(documents)

    def invoke(
        self,
        user_query: str,
        system_prompt: str | None = None,
        forced_context: list[str] | None = None,
        retrieve_only: bool = False,
    ) -> TargetResponse:
        """
        Executes the RAG pipeline for a single input user query.

        Controls the flow of data through Retrieval, Defense (input), Prompt Assembly,
        Generation, and Defense (output).

        Args:
            user_query: The end-user's query.
            system_prompt: Optional override for the system instruction.
                           If None, uses the one loaded based on the config.
            forced_context: List of strings to use as context, bypassing the retriever.
                            Useful for testing Generator robustness in isolation.
            retrieve_only: If True, executes only the retrieval step and returns the
                           found context in metadata, skipping LLM generation.

        Returns:
            TargetResponse containing model output (or empty string if retrieve_only),
            metadata about the run (context, model), and any defense triggers/signals.
        """

        original_system_prompt = system_prompt or self.system_prompt

        # Retrieval step
        context_chunks = []

        if forced_context is not None:
            context_chunks = forced_context
            logger.debug("Using forced context.")
        elif self.vector_store:
            context_chunks = self.vector_store.search(user_query)
            logger.debug(f"Retrieved {len(context_chunks)} chunks.")

        # For Retriever only testing
        if retrieve_only:
            return TargetResponse(
                content="", raw_content=None, used_context=context_chunks
            )

        # Defense: Canary injection (input side)
        canary_token = None
        system_prompt_with_canary = original_system_prompt
        if self.canary_enabled:
            # SDK modifies the system prompt to include the hidden token instructions
            system_prompt_with_canary, canary_token = self.canary.inject(
                original_system_prompt
            )
            logger.debug("Canary token injected into system prompt.")

        formatted_request_prompt = self.prompt_template.format(
            query=user_query, context=context_chunks
        )

        # Generation
        if not self.llm:
            logger.error("Invoke called but no LLM is configured.")
            # We return a response indicating failure in content, or raise error?
            # Returning error message in content is safer for the runner loop.
            return TargetResponse(
                content="Error: No LLM Configured", used_context=context_chunks
            )

        raw_response: str | None = self.llm.generate(
            system_message=system_prompt_with_canary,
            user_message=formatted_request_prompt,
        )

        if not raw_response:
            raise ValueError("LLM response is not a valid string!")

        # Defense: Canary check (output side)
        attack_detected = False
        detection_reason = None
        final_content = raw_response

        if self.canary_enabled and canary_token:
            result: CanaryResult = self.canary.check(raw_response, canary_token)

            # Map SDK result directly to response model
            if result.threat_detected:
                attack_detected = True
                detection_reason = "Canary Integrity Check Failed"
                final_content = "Response blocked by Deconvolute."
            else:
                # If safe, we clean the token from the final output
                final_content = self.canary.clean(raw_response, canary_token)

        # Metadata preparation
        response_metadata = {
            "model": self.config.llm.model if self.config.llm else "none",
        }

        if canary_token:
            response_metadata["canary_token"] = canary_token

        return TargetResponse(
            content=final_content,
            raw_content=raw_response,
            used_context=context_chunks,
            attack_detected=attack_detected,
            detection_reason=detection_reason,
            metadata=response_metadata,
        )
