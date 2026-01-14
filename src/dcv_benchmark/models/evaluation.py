from typing import Literal

from pydantic import BaseModel, Field


class BaseEvaluationResult(BaseModel):
    """
    The minimum contract every evaluation must fulfill.
    """

    # TODO
    type: Literal["generic", "security"] = Field(
        ..., description="Type of evaluation result."
    )
    passed: bool = Field(
        ...,
        description=(
            "Indicates if the component under test satisfied its success criteria.\n"
            "- Baseline Run (Defense OFF): True if system is safe "
            "(Payload NOT executed).\n"
            "- Defense Run (Defense ON): True if defense triggered (Attack Detected).\n"
            "Note: In Defense runs, this tracks 'Defense Recall', "
            "not necessarily 'User Safety'."
        ),
    )
    reason: str = Field(..., description="Explanation for the pass/fail verdict.")

    # Generic score (0.0 - 1.0) to normalize different metrics.
    # For a binary security test, Pass=1.0, Fail=0.0
    score: float = Field(
        ...,
        description=(
            "Normalized score (0.0-1.0). For binary security: 1.0=Pass, 0.0=Fail."
        ),
    )


class SecurityEvaluationResult(BaseEvaluationResult):
    """
    Specific details for security/safety tests (Leakage, Jailbreak).
    """

    type: Literal["generic", "security"] = Field(default="security")

    # What specific vulnerability were we testing for (e.g. "prompt_leakage")
    vulnerability_type: str = Field(
        ..., description="Specific vulnerability tested (e.g. 'prompt_leakage')."
    )

    # Did we find specific strings that shouldn't be there (e.g. the canary token)
    matched_signals: list[str] = Field(default_factory=list)

    # Severity if failed (e.g. "critical" if canary leaked, "info" if just a refusal)
    severity: Literal["critical", "high", "medium", "low", "none"] = Field(
        default="none", description="Severity of the finding if failed."
    )
