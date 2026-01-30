from typing import Any, Literal

from pydantic import BaseModel, Field

from dcv_benchmark.models.config.defense import DefenseConfig


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "mock"] = Field(..., description="Embedding provider.")
    model: str = Field(..., description="Model name.")


class RetrieverConfig(BaseModel):
    provider: Literal["chroma", "mock"] = Field(..., description="Retriever provider.")
    top_k: int = Field(default=3, description="Number of chunks to retrieve.")
    chunk_size: int = Field(default=500, description="Size of text chunks.")


class LLMConfig(BaseModel):
    provider: Literal["openai"] = Field(..., description="LLM provider.")
    model: str = Field(..., description="Model name.")
    temperature: float = Field(default=0.0, description="Sampling temperature.")


class SystemPromptConfig(BaseModel):
    """Developer-provided system prompt"""

    file: str = Field(..., description="Name of prompt file.")
    key: str = Field(..., description="Key within the prompts file.")


class PromptTemplateConfig(BaseModel):
    """Template with placeholders for user and context."""

    file: str = Field(..., description="Name of templates file.")
    key: str = Field(..., description="Key within the templates file.")


class TargetConfig(BaseModel):
    name: str = Field(..., description="Pipeline type (e.g. basic_rag).")
    system_prompt: SystemPromptConfig = Field(..., description="System prompt config.")
    prompt_template: PromptTemplateConfig = Field(..., description="Template config.")
    defense: DefenseConfig = Field(..., description="Defense configuration.")
    generate: bool = Field(
        default=True,
        description=(
            "If False, stops execution after input defenses (Simulated Scan Mode)."
        ),
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
