from typing import Literal

from pydantic import BaseModel, Field

from dcv_benchmark.models.config.target import TargetConfig


class SquadInputConfig(BaseModel):
    type: Literal["squad"] = Field(..., description="Type of dataset.")
    dataset_name: str = Field(
        ..., description="Name of the dataset (e.g. 'squad_canary_v1')"
    )


class BipiaInputConfig(BaseModel):
    type: Literal["bipia"] = Field(..., description="Type of dataset.")
    tasks: list[Literal["email", "code", "table", "qa"]] = Field(
        ..., description="BIPIA tasks to generate."
    )
    injection_pos: Literal["start", "middle", "end"] = Field(
        default="end", description="Position of the injection."
    )
    max_samples: int | None = Field(
        default=None, description="Maximum number of samples to generate."
    )
    seed: int = Field(default=42, description="Random seed.")


InputConfig = SquadInputConfig | BipiaInputConfig


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

    input: InputConfig = Field(..., description="Input data configuration.")
    target: TargetConfig = Field(..., description="Target system configuration.")
    scenario: ScenarioConfig = Field(..., description="Scenario configuration.")

    evaluator: EvaluatorConfig | None = Field(
        default=None, description="Explicit evaluator configuration."
    )

    model_config = {"extra": "forbid"}
