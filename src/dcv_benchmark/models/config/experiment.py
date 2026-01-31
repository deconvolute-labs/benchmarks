from pydantic import BaseModel, Field

from dcv_benchmark.models.config.target import LLMConfig, TargetConfig


# The full experiment config
class ExperimentConfig(BaseModel):
    name: str = Field(..., description="Name of the experiment.")
    description: str = Field(default="", description="Description of the experiment.")
    version: str = Field(default="N/A", description="Version of the experiment.")

    # Dataset directory name (e.g. "squad_val", "bipia_val")
    dataset: str = Field(
        ...,
        description="Name of the compiled dataset folder in workspace/datasets/built.",
    )

    # Main target config
    target: TargetConfig = Field(..., description="Target system configuration.")

    # Optional dedicated Judge LLM
    judge_llm: LLMConfig | None = Field(
        default=None,
        description="Dedicated LLM configuration for evaluators/judges.",
    )

    model_config = {"extra": "forbid"}
