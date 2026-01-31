from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from dcv_benchmark.models.evaluation import (
    BaseEvaluationResult,
    SecurityEvaluationResult,
)
from dcv_benchmark.models.responses import TargetResponse


class TraceItem(BaseModel):
    """
    Represents the full execution lifecycle of a single test sample.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    latency_seconds: float = Field(
        ..., description="Time taken for the target to respond"
    )

    # Link back to the static dataset
    sample_id: str = Field(..., description="ID from the original dataset")
    sample_type: Literal["attack", "benign"]
    attack_strategy: str = Field(
        default="none",
        description="The attack strategy used, or 'none'.",
    )

    # The user input for this specific run
    user_query: str | None = None

    # The full execution result (contains output + used_context + defense signals)
    response: TargetResponse

    # The score/grade per evaluator
    evaluations: dict[str, SecurityEvaluationResult | BaseEvaluationResult] = Field(
        default_factory=dict
    )
