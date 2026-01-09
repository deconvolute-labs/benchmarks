from typing import Any

from pydantic import BaseModel, Field


class TargetResponse(BaseModel):
    """
    Standardized output from a Target pipeline.

    Can represent:
    1. Full RAG: content + retrieved_context
    2. Generator Only: content (empty context)
    3. Retriever Only: retrieved_context (content is None)
    """

    # The final output the user sees (potentially sanitized)
    # Optional because a Retriever-only run produces no generated text
    content: str | None = None

    # The direct output from the Generator/LLM before defense/sanitization
    # Essential for analyzing if the model was hijacked even if the output was blocked
    raw_content: str | None = None

    # Context
    used_context: list[str] = Field(default_factory=list)

    # Defense signals
    attack_detected: bool = Field(
        default=False, description="True if an integrity check failed"
    )
    detection_reason: str | None = Field(
        default=None, description="Reason for detection (e.g. 'canary_missing')."
    )

    # Catch-all for extra info (token usage, latency, model names, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)
