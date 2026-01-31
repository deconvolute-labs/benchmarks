from typing import Any

from pydantic import BaseModel, Field

from dcv_benchmark.models.config.target import TargetConfig


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

    target: TargetConfig = Field(..., description="Target system configuration.")

    # Evaluators: Key is evaluator name (e.g., 'keyword'), Value is its settings
    evaluators: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Dictionary of evaluators and their settings."
    )

    model_config = {"extra": "forbid"}
