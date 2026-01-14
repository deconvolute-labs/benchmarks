from pydantic import BaseModel, Field


class StrategySecurityMetric(BaseModel):
    """
    Metrics for a specific attack strategy (e.g. 'leet_speak').

    We track how many samples were used and the Attack Success Rate (ASV).
    We also expose raw counts to allow plotting detection rates.
    """

    samples: int = Field(..., description="Number of samples")
    asv: float = Field(..., description="Attack Success Value")
    detected_count: int = Field(..., description="True Positives (Attacks caught)")
    missed_count: int = Field(..., description="False Negatives (Attacks successful)")


class GlobalSecurityMetrics(BaseModel):
    """Aggregate metrics for the entire run."""

    total_samples: int = Field(..., description="Total number of samples processed")
    asv_score: float = Field(..., description="Attack Success Value (ASV)")
    pna_score: float = Field(..., description="Performance No Attack (PNA)")
    fpr_score: float = Field(..., description="False Positive Rate (FPR)")

    tp: int = Field(..., description="True Positives (Attacks caught)")
    fn: int = Field(..., description="False Negatives (Silent failures)")
    tn: int = Field(..., description="True Negatives (Benign passed)")
    fp: int = Field(..., description="False Positives (False alarms)")

    avg_latency_seconds: float = Field(
        ..., description="Average latency per request in seconds"
    )

    # Raw Data for Plotting (we exclude from repr if possible, but keep simple for now)
    latencies_attack: list[float] = Field(
        default_factory=list, description="Raw latencies for attack samples"
    )
    latencies_benign: list[float] = Field(
        default_factory=list, description="Raw latencies for benign samples"
    )


class SecurityMetrics(BaseModel):
    """The root container for a Security Experiment's results."""

    type: str = Field(default="security", description="Type of metric.")
    global_metrics: GlobalSecurityMetrics = Field(
        ..., description="Aggregated metrics for the whole run."
    )
    by_strategy: dict[str, StrategySecurityMetric] = Field(
        ..., description="Metrics broken down by attack strategy."
    )
