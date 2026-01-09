from typing import Literal

from pydantic import BaseModel, Field


class BaseEvaluationResult(BaseModel):
    """
    The minimum contract every evaluation must fulfill.
    """

    type: Literal["generic", "security"]
    passed: bool
    reason: str

    # Generic score (0.0 - 1.0) to normalize different metrics.
    # For a binary security test, Pass=1.0, Fail=0.0
    score: float


class SecurityEvaluationResult(BaseEvaluationResult):
    """
    Specific details for security/safety tests (Leakage, Jailbreak).
    """

    type: Literal["generic", "security"] = "security"

    # What specific vulnerability were we testing for (e.g. "prompt_leakage")
    vulnerability_type: str

    # Did we find specific strings that shouldn't be there (e.g. the canary token)
    matched_signals: list[str] = Field(default_factory=list)

    # Severity if failed (e.g. "critical" if canary leaked, "info" if just a refusal)
    severity: Literal["critical", "high", "medium", "low", "none"] = "none"
