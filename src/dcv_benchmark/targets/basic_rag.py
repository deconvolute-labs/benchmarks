from deconvolute import (
    CanaryDetector,
    LanguageDetector,
)
from deconvolute.detectors.content.language.models import LanguageResult
from deconvolute.detectors.content.signature.engine import SignatureDetector
from deconvolute.detectors.integrity.canary.models import CanaryResult

from dcv_benchmark.components.llms import BaseLLM, create_llm
from dcv_benchmark.components.vector_store import create_vector_store
from dcv_benchmark.models.experiments_config import TargetConfig
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.targets.base import BaseTarget
from dcv_benchmark.utils.logger import get_logger
from dcv_benchmark.utils.prompt_loader import load_prompt_text

logger = get_logger(__name__)


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
        # 1. Canary Defense (LLM Input/Output Layer)
        self.canary = CanaryDetector()
        self.canary_enabled = False
        if config.defense.canary and config.defense.canary.enabled:
            self.canary_enabled = True
            logger.info(
                f"Defense [Canary]: ENABLED. Settings: {config.defense.canary.settings}"
            )

        # 2. Language Defense (Output Layer)
        self.language_detector: LanguageDetector | None = None
        if config.defense.language and config.defense.language.enabled:
            self.language_detector = LanguageDetector(
                **config.defense.language.settings
            )
            logger.info(
                "Defense [Language]: ENABLED. Config: "
                f"{config.defense.language.settings}"
            )

        # 3. Signature Defense (Ingestion Layer - YARA)
        self.signature_detector: SignatureDetector | None = None
        if config.defense.yara and config.defense.yara.enabled:
            self.signature_detector = SignatureDetector(**config.defense.yara.settings)
            logger.info(
                "Defense [Signature/YARA]: ENABLED. Config: "
                f"{config.defense.yara.settings}"
            )

        # Load system prompt
        self.system_prompt: str = load_prompt_text(
            path=config.system_prompt.file,
            key=config.system_prompt.key,
        )

        # Load prompt template
        self.prompt_template: str = load_prompt_text(
            path=config.prompt_template.file,
            key=config.prompt_template.key,
        )

    def ingest(self, documents: list[str]) -> None:
        """
        Populates the vector store with the provided dataset.

        This method is idempotent-ish for the benchmark run (adds to the ephemeral DB).
        If no vector store is configured, this operation logs a warning and skips.
        Applies Ingestion-side defenses (YARA, ML) if enabled.

        Args:
            documents: A list of text strings (knowledge base) to index.
        """
        if not self.vector_store:
            logger.warning("Ingest called but no Vector Store is configured. Skipping.")
            return

        safe_documents = []
        blocked_count = 0
        total_docs = len(documents)

        logger.info(f"Starting ingestion scan for {total_docs} documents...")

        for doc in documents:
            is_clean = True

            # Check 1: Signature / YARA
            if self.signature_detector:
                result = self.signature_detector.check(doc)
                if result.threat_detected:
                    is_clean = False
                    logger.debug(
                        "Doc blocked by SignatureDetector: "
                        f"{getattr(result, 'metadata', 'N/A')}"
                    )

            if is_clean:
                safe_documents.append(doc)
            else:
                blocked_count += 1

        logger.info(
            f"Ingestion Scan Complete: {len(safe_documents)} accepted, "
            f"{blocked_count} blocked (Threats)."
        )

        if safe_documents:
            self.vector_store.add_documents(safe_documents)

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
            # If we have a Signature Detector (YARA/Scanner),
            # we check the raw docs here.
            if self.signature_detector:
                for chunk in forced_context:
                    scan_result = self.signature_detector.check(chunk)

                    if scan_result.threat_detected:
                        # HIT: Threat detected on raw document.
                        # We STOP here. No LLM call.
                        logger.info(
                            "Scan Defense triggered on raw context: "
                            f"{scan_result.metadata}"
                        )
                        return TargetResponse(
                            content="[Blocked by Signature Scan]",
                            raw_content=None,
                            used_context=forced_context,
                            attack_detected=True,
                            detection_reason=(
                                f"Signature Scan: "
                                f"{getattr(scan_result, 'metadata', 'Threat')}"
                            ),
                            metadata={"stage": "ingestion_scan"},
                        )

            # If we get here, the Scan missed (or no scanner enabled).
            context_chunks = forced_context
            logger.debug("Using forced context (Simulated Ingestion).")
        elif self.vector_store:
            context_chunks = self.vector_store.search(user_query)
            logger.debug(f"Retrieved {len(context_chunks)} chunks.")

        # 2. Check Generation Flag (The "Scan Mode" Support)
        # If the user configured generate=False, we stop here.
        # This covers the "Miss" case where we don't want to waste tokens on the LLM.
        if not self.config.generate or retrieve_only:
            return TargetResponse(
                content="",  # Empty content
                raw_content=None,
                used_context=context_chunks,
                attack_detected=False,  # We scanned, but found nothing
                detection_reason=None,
                metadata={"stage": "ingestion_scan", "skipped_generation": True},
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

        # Metadata preparation
        response_metadata = {
            "model": self.config.llm.model if self.config.llm else "none",
        }

        if canary_token:
            response_metadata["canary_token"] = canary_token

        # Layer A: Canary Check
        if self.canary_enabled and canary_token:
            result: CanaryResult = self.canary.check(raw_response, token=canary_token)

            if result.threat_detected:
                attack_detected = True
                detection_reason = "Canary Integrity Check Failed"
                final_content = "Response blocked by Deconvolute."
            else:
                # If safe, clean the token before passing to next layer
                final_content = self.canary.clean(raw_response, canary_token)

        # Layer B: Language Check (Daisy Chained)
        # We only run this if the previous layer didn't block it
        if not attack_detected and self.language_detector:
            # We pass reference_text to enable Mode B if the detector supports it
            lang_result: LanguageResult = self.language_detector.check(
                content=final_content, reference_text=user_query
            )

            # Store result in metadata for debugging/analysis
            # Using dict() or model_dump() depending on Pydantic version in SDK
            response_metadata["language_check"] = (
                lang_result.model_dump()
                if hasattr(lang_result, "model_dump")
                else lang_result.__dict__
            )

            if lang_result.threat_detected:
                attack_detected = True
                detection_reason = (
                    f"Language Policy Violation: {lang_result.detected_language}"
                )
                final_content = "Response blocked by Deconvolute."

        return TargetResponse(
            content=final_content,
            raw_content=raw_response,
            used_context=context_chunks,
            attack_detected=attack_detected,
            detection_reason=detection_reason,
            metadata=response_metadata,
        )
