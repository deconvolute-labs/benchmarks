import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from dcv_benchmark.models.metrics import SecurityMetrics


class ReportMeta(BaseModel):
    """Metadata regarding the execution context of the experiment."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    timestamp_start: datetime
    timestamp_end: datetime
    duration_seconds: float
    deconvolute_version: str = "N/A"
    runner_version: str = "1.0.0"


class ExperimentReport(BaseModel):
    """
    The root schema for the 'results.json' file.
    """

    meta: ReportMeta
    config: dict[str, Any]
    metrics: SecurityMetrics | dict[str, Any]
