from typing import Any, Literal

from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset JSON file")


class DefenseLayerConfig(BaseModel):
    type: str
    enabled: bool
    settings: dict[str, Any] = Field(default_factory=dict)


class DefenseConfig(BaseModel):
    type: Literal["deconvolute", "none"] = "deconvolute"
    required_version: str | None = None
    layers: list[DefenseLayerConfig] = Field(default_factory=list)


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "mock"]
    model: str


class RetrieverConfig(BaseModel):
    provider: Literal["chroma", "mock"]
    top_k: int = 3
    chunk_size: int = 500


class LLMConfig(BaseModel):
    provider: Literal["openai"]
    model: str
    temperature: float = 0.0


class SystemPromptConfig(BaseModel):
    """Developer-provided system prompt"""

    path: str
    key: str


class PromptTemplateConfig(BaseModel):
    """Template with placeholders for user and context."""

    path: str
    key: str


class TargetConfig(BaseModel):
    pipeline: str
    system_prompt: SystemPromptConfig
    prompt_template: PromptTemplateConfig
    defense: DefenseConfig
    embedding: EmbeddingConfig | None = None
    retriever: RetrieverConfig | None = None
    llm: LLMConfig | None = None
    pipeline_params: dict[str, Any] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    id: str


# The full experiment config
class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    version: str = "N/A"

    input: InputConfig
    target: TargetConfig
    scenario: ScenarioConfig

    # Strict validation: Throw error if unknown keys appear in YAML
    model_config = {"extra": "forbid"}
