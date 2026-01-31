from typing import Any, Literal

from pydantic import BaseModel, Field

from dcv_benchmark.models.config.defense import DefenseConfig


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "mock"] = Field(..., description="Embedding provider.")
    model: str = Field(..., description="Model name.")


class RetrieverConfig(BaseModel):
    provider: Literal["chromadb", "mock"] = Field(
        ..., description="Retriever provider."
    )
    k: int = Field(default=3, description="Number of chunks to retrieve.")
    chunk_size: int = Field(default=500, description="Size of text chunks.")


class LLMConfig(BaseModel):
    provider: Literal["openai"] = Field(..., description="LLM provider.")
    model: str = Field(..., description="Model name.")
    temperature: float = Field(default=0.0, description="Sampling temperature.")


class SystemPromptConfig(BaseModel):
    """Developer-provided system prompt"""

    file: str | None = Field(default=None, description="Name of prompt file.")
    key: str = Field(..., description="Key within the prompts file.")


class PromptTemplateConfig(BaseModel):
    """Template with placeholders for user and context."""

    file: str | None = Field(default=None, description="Name of templates file.")
    key: str = Field(..., description="Key within the templates file.")


class TargetConfig(BaseModel):
    name: str = Field(..., description="Pipeline type (e.g. basic_rag).")

    # Execution Control
    generate: bool = Field(
        default=True,
        description=(
            "If False, stops execution after input defenses (Simulated Scan Mode)."
        ),
    )

    # Defenses
    defense: DefenseConfig = Field(
        default_factory=DefenseConfig, description="Defense configuration."
    )

    # Components (Optional to allow defaults or skip)
    system_prompt: SystemPromptConfig | None = Field(
        default=None, description="System prompt config."
    )
    prompt_template: PromptTemplateConfig | None = Field(
        default=None, description="Template config."
    )

    embedding: EmbeddingConfig | None = Field(
        default=None, description="Embedding config."
    )
    retriever: RetrieverConfig | None = Field(
        default=None, description="Retriever config."
    )
    llm: LLMConfig | None = Field(default=None, description="LLM configuration.")

    pipeline_params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
