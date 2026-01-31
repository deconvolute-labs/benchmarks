from deconvolute import ThreatDetectedError, guard

from dcv_benchmark.components.llms import BaseLLM, OpenAILLM, create_llm
from dcv_benchmark.components.vector_store import create_vector_store
from dcv_benchmark.models.experiments_config import TargetConfig
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.targets.base import BaseTarget
from dcv_benchmark.utils.logger import get_logger
from dcv_benchmark.utils.prompt_loader import load_prompt_text

logger = get_logger(__name__)


class BasicRAGGuard(BaseTarget):
    """
    A RAG implementation that uses the Deconvolute 'guard()' orchestrator API.

    Unlike BasicRAG (which manually layers detectors), this target wraps the
    LLM client directly, delegating defense logic to the SDK's defaults.
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

            # Apply Guard Wrapper
            # We must wrap the internal client of the LLM adapter.
            # In BasicRAGGuard, we assume we want to use the Deconvolute guard.
            # We can optionally check if any detector is enabled, but guard()
            # handles config internally usually.
            # For now, we wrap it unconditionally if it's BasicRAGGuard.
            # Let's check if any detector is enabled to be safe, or just wrap it.
            # The SDK guard() might need config passed to it or it picks up
            # from env/defaults?
            # Assuming unconditional wrap for this target type is the intended
            # behavior for BasicRAGGuard.

            if isinstance(self.llm, OpenAILLM):
                logger.info("Deconvolute Guard: Wrapping OpenAI Client.")
                # guard() returns a wrapped client that mimics the OpenAI interface
                self.llm.client = guard(self.llm.client)
            else:
                logger.warning(
                    "Deconvolute Guard is enabled but LLM provider "
                    f"'{config.llm.provider}' is not automatically supported by "
                    "this benchmark adapter."
                )

        # Setup vector store
        self.vector_store = None
        if config.embedding and config.retriever:
            self.vector_store = create_vector_store(config.retriever, config.embedding)
            logger.debug("Vector Store initialized.")
        else:
            logger.debug("No Retriever configured. Running in Generator-only mode.")

        # Load system prompt
        sys_file = config.system_prompt.file if config.system_prompt else None
        sys_key = config.system_prompt.key if config.system_prompt else "standard"

        self.system_prompt: str = load_prompt_text(
            path=sys_file or "prompts/system_prompts.yaml",
            key=sys_key,
        )

        # Load prompt template
        tpl_file = config.prompt_template.file if config.prompt_template else None
        tpl_key = (
            config.prompt_template.key if config.prompt_template else "rag_default"
        )

        self.prompt_template: str = load_prompt_text(
            path=tpl_file or "prompts/templates.yaml",
            key=tpl_key,
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

        # Prompt Assembly
        formatted_request_prompt = self.prompt_template.format(
            query=user_query, context=context_chunks
        )

        # Generation
        if not self.llm:
            logger.error("Invoke called but no LLM is configured.")
            # Returning error message in content is safer for the runner loop.
            return TargetResponse(
                content="Error: No LLM Configured", used_context=context_chunks
            )

        # Try-Catch
        try:
            # If the client is wrapped, guard() logic executes here.
            # It handles injection (input) and checking (output) internally.
            raw_response: str | None = self.llm.generate(
                system_message=original_system_prompt,
                user_message=formatted_request_prompt,
            )

            # If we reach here, no threat was detected by the guard
            if not raw_response:
                raise ValueError("LLM response is not a valid string!")

            return TargetResponse(
                content=raw_response,
                raw_content=raw_response,
                used_context=context_chunks,
                attack_detected=False,
                detection_reason=None,
                metadata={
                    "strategy": "guard",
                    "model": self.config.llm.model if self.config.llm else None,
                },
            )

        except ThreatDetectedError as e:
            # The Guard API detected a threat and blocked execution
            logger.info(f"Threat detected by Deconvolute Guard: {e}")

            return TargetResponse(
                content="Response blocked by Deconvolute.",
                raw_content=None,  # Content was blocked
                used_context=context_chunks,
                attack_detected=True,
                detection_reason=str(e),
                metadata={
                    "strategy": "guard",
                    "model": self.config.llm.model if self.config.llm else None,
                },
            )
