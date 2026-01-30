from typing import Literal

from pydantic import BaseModel, Field

from dcv_benchmark.models.config.target import TargetConfig


class InputConfig(BaseModel):
    dataset_name: str | None = Field(
        default=None, description="Name of the dataset (e.g. 'squad_canary_v1')"
    )


class EvaluatorConfig(BaseModel):
    type: Literal["canary", "keyword", "language_mismatch", "bipia"] = Field(
        ..., description="Type of evaluator to use."
    )
    # For language_mismatch
    expected_language: str = Field(
        default="en", description="Expected language ISO code (e.g. 'en')."
    )
    strict: bool = Field(
        default=True, description="If True, minor deviations cause failure."
    )
    # For keyword (optional override)
    target_keyword: str | None = Field(
        default=None, description="Override the default target keyword."
    )


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

    evaluator: EvaluatorConfig | None = Field(
        default=None, description="Explicit evaluator configuration."
    )

    model_config = {"extra": "forbid"}
