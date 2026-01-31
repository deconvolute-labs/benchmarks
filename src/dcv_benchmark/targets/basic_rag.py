from typing import Any, Literal, cast

from deconvolute import (
    CanaryDetector,
    LanguageDetector,
)
from deconvolute.detectors.content.language.models import LanguageResult
from deconvolute.detectors.content.signature.engine import SignatureDetector
from deconvolute.detectors.integrity.canary.models import CanaryResult

from dcv_benchmark import defaults
from dcv_benchmark.components.llms import BaseLLM, create_llm
from dcv_benchmark.components.vector_store import create_vector_store
from dcv_benchmark.models.config.target import LLMConfig, TargetConfig
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

        # 1. Initialization Logic (Lazy Loading based on 'generate' flag)
        self.llm: BaseLLM | None = None
        self.vector_store: Any | None = None
        self.system_prompt: str | None = None
        self.prompt_template: str | None = None

        if config.generate:
            self._init_generation_components(config)
        else:
            logger.info(
                "Target [basic_rag]: Running in SCAN MODE (Generation Disabled)."
            )

        # 2. Defense Setup (Nested Stages)
        self._init_defenses(config)

    def _init_generation_components(self, config: TargetConfig) -> None:
        """Initializes LLM, Retriever, and Prompts using defaults if necessary."""

        # A. LLM
        llm_config = config.llm
        if not llm_config:
            logger.info(
                f"No LLM config provided. Using defaults: {defaults.DEFAULT_LLM_MODEL}"
            )
            llm_config = LLMConfig(
                provider=cast(Literal["openai"], defaults.DEFAULT_LLM_PROVIDER),
                model=defaults.DEFAULT_LLM_MODEL,
                temperature=defaults.DEFAULT_LLM_TEMPERATURE,
            )
            # Update config for reporting (Effective Config)
            self.config.llm = llm_config

        logger.debug(f"Initializing LLM: {llm_config.provider} ({llm_config.model})")
        self.llm = create_llm(llm_config)

        # B. Vector Store (Retriever + Embeddings)
        # We need both to support retrieval.
        if config.embedding and config.retriever:
            self.vector_store = create_vector_store(config.retriever, config.embedding)
            logger.debug("Vector Store initialized.")
        elif config.retriever:
            # Only retriever provided, not handled yet.
            pass

        # C. Prompts
        # System Prompt
        sys_key = (
            config.system_prompt.key
            if config.system_prompt
            else defaults.DEFAULT_SYSTEM_PROMPT_KEY
        )
        sys_file = (
            config.system_prompt.file
            if config.system_prompt
            else "prompts/system_prompts.yaml"
        )
        self.system_prompt = load_prompt_text(
            path=sys_file or "prompts/system_prompts.yaml", key=sys_key
        )

        # Template
        tpl_key = (
            config.prompt_template.key
            if config.prompt_template
            else defaults.DEFAULT_TEMPLATE_KEY
        )
        tpl_file = (
            config.prompt_template.file
            if config.prompt_template
            else "prompts/templates.yaml"
        )
        self.prompt_template = load_prompt_text(
            path=tpl_file or "prompts/templates.yaml", key=tpl_key
        )

    def _init_defenses(self, config: TargetConfig) -> None:
        """Initializes defenses for ingestion and generation stages."""

        # Stage 1: Ingestion
        ingestion = config.defense.ingestion

        # Signature Detector
        self.signature_detector: SignatureDetector | None = None
        if ingestion.signature_detector.enabled:
            # Pass **settings to override defaults
            self.signature_detector = SignatureDetector(
                **ingestion.signature_detector.settings
            )
            logger.info("Defense [Ingestion/Signature]: ENABLED")

        # Stage 2: Generation
        generation = config.defense.generation

        # Canary Detector
        self.canary: CanaryDetector | None = None
        if generation.canary_detector.enabled:
            self.canary = CanaryDetector(**generation.canary_detector.settings)
            logger.info("Defense [Generation/Canary]: ENABLED")

        # Language Detector
        self.language_detector: LanguageDetector | None = None
        if generation.language_detector.enabled:
            self.language_detector = LanguageDetector(
                **generation.language_detector.settings
            )
            logger.info("Defense [Generation/Language]: ENABLED")

    def _run_ingestion_checks(self, documents: list[str]) -> bool:
        """
        Runs ingestion-stage defenses (Signature) on a list of raw documents.
        Returns True if ANY threat is detected (Blocked).
        """
        if not documents:
            return False

        # Signature Check
        if self.signature_detector:
            for doc in documents:
                result = self.signature_detector.check(doc)
                if result.threat_detected:
                    logger.info(
                        f"Blocked by Signature: {getattr(result, 'metadata', '')}"
                    )
                    return True

        return False

    def ingest(self, documents: list[str]) -> None:
        """
        Populates the target's vector store with the provided corpus.
        Filters out blocked documents during ingestion.
        """
        if not self.vector_store:
            logger.warning("Ingest called but no Vector Store is configured. Skipping.")
            return

        safe_documents = []
        blocked_count = 0

        logger.info(f"Starting ingestion scan for {len(documents)} documents ...")

        for doc in documents:
            # run_ingestion_checks returns True if BLOCKED
            if self._run_ingestion_checks([doc]):
                blocked_count += 1
            else:
                safe_documents.append(doc)

        logger.info(
            f"Ingestion Scan Complete: {len(safe_documents)} accepted, "
            f"{blocked_count} blocked."
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
        # Context Retrieval / Resolution
        context_chunks = []
        used_context = []

        if forced_context is not None:
            # When using forced_context, we treat it as "Ingestion" time for the check.
            # E.g. simulating that these docs are entering the system.
            if self._run_ingestion_checks(forced_context):
                return TargetResponse(
                    content="[Blocked by Ingestion Defenses]",
                    raw_content=None,
                    used_context=forced_context,
                    attack_detected=True,
                    detection_reason="Ingestion/Signature Block",
                    metadata={"stage": "ingestion"},
                )
            context_chunks = forced_context
            used_context = forced_context
            logger.debug("Using forced context (Simulated Ingestion).")

        elif self.vector_store:
            # If standard retrieval, we assume ingestion checks happened at
            # ingest() time.
            context_chunks = self.vector_store.search(user_query)
            used_context = context_chunks

        # Check Execution Mode
        # If generate=False, we stop here (Scan Mode Simulation)
        if not self.config.generate or retrieve_only:
            return TargetResponse(
                content="",
                raw_content=None,
                used_context=used_context,
                attack_detected=False,
                detection_reason=None,
                metadata={"stage": "scan", "skipped_generation": True},
            )

        # Prompt Assembly & Canary Injection
        effective_sys_prompt = system_prompt or self.system_prompt or ""
        canary_token = None

        if self.canary:
            effective_sys_prompt, canary_token = self.canary.inject(
                effective_sys_prompt
            )

        formatted_prompt = ""
        if self.prompt_template:
            formatted_prompt = self.prompt_template.format(
                query=user_query, context=context_chunks
            )
        else:
            # Fallback if no template (shouldn't happen with defaults)
            logger.info("No prompt template provided. Using fallback ...")
            formatted_prompt = f"{user_query}\n\nContext:\n{context_chunks}"

        # Generation
        if not self.llm:
            return TargetResponse(
                content="Error: No LLM Configured", used_context=used_context
            )

        raw_response = self.llm.generate(
            system_message=effective_sys_prompt, user_message=formatted_prompt
        )
        if not raw_response:
            raise ValueError("LLM response is not a valid string!")

        # Generation Defenses (Output Side)
        final_content = raw_response
        attack_detected = False
        reason = None
        metadata: dict[str, Any] = {
            "model": self.llm.config.model if self.llm else "unknown"
        }

        # Canary Check
        if self.canary and canary_token:
            metadata["canary_token"] = canary_token
            c_result: CanaryResult = self.canary.check(raw_response, token=canary_token)
            if c_result.threat_detected:
                attack_detected = True
                reason = "Canary Integrity Check Failed"
                final_content = "Response blocked by Deconvolute."
            else:
                final_content = self.canary.clean(raw_response, canary_token)

        # Language Check
        if not attack_detected and self.language_detector:
            l_result: LanguageResult = self.language_detector.check(
                content=final_content, reference_text=user_query
            )
            if hasattr(l_result, "model_dump"):
                metadata["language_check"] = l_result.model_dump()

            if l_result.threat_detected:
                attack_detected = True
                reason = f"Language Violation: {l_result.detected_language}"
                final_content = "Response blocked by Deconvolute."

        return TargetResponse(
            content=final_content,
            raw_content=raw_response,
            used_context=context_chunks,
            attack_detected=attack_detected,
            detection_reason=reason,
            metadata=metadata,
        )
