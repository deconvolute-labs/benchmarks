from typing import Any, Literal

from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    dataset_path: str | None = Field(
        default=None, description="Path to the dataset JSON file"
    )


class DefenseLayerConfig(BaseModel):
    type: str = Field(..., description="Type of defense layer.")
    enabled: bool = Field(..., description="Whether this layer is active.")
    settings: dict[str, Any] = Field(default_factory=dict)


class DefenseConfig(BaseModel):
    type: Literal["deconvolute", "none"] = Field(
        default="deconvolute", description="Defense provider."
    )
    required_version: str | None = Field(
        default=None, description="Min version required."
    )
    layers: list[DefenseLayerConfig] = Field(default_factory=list)


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
    pipeline: str = Field(..., description="Pipeline type (e.g. basic_rag).")
    system_prompt: SystemPromptConfig = Field(..., description="System prompt config.")
    prompt_template: PromptTemplateConfig = Field(..., description="Template config.")
    defense: DefenseConfig = Field(..., description="Defense configuration.")
    embedding: EmbeddingConfig | None = Field(
        default=None, description="Embedding config."
    )
    retriever: RetrieverConfig | None = Field(
        default=None, description="Retriever config."
    )
    llm: LLMConfig | None = Field(default=None, description="LLM configuration.")
    pipeline_params: dict[str, Any] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    id: str = Field(..., description="Scenario ID.")


# The full experiment config
class ExperimentConfig(BaseModel):
    name: str = Field(..., description="Name of the experiment.")
    description: str = Field(default="", description="Description of the experiment.")
    version: str = Field(default="N/A", description="Version of the experiment.")

    input: InputConfig = Field(
        default_factory=InputConfig, description="Input data configuration."
    )
    target: TargetConfig = Field(..., description="Target system configuration.")
    scenario: ScenarioConfig = Field(..., description="Scenario configuration.")

    # Strict validation: Throw error if unknown keys appear in YAML
    model_config = {"extra": "forbid"}
