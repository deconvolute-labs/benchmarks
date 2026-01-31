from typing import Any

from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether the detector is enabled.")
    settings: dict[str, Any] = Field(
        default_factory=dict, description="Detector-specific settings."
    )


class IngestionStageConfig(BaseModel):
    signature_detector: DetectorConfig = Field(default_factory=DetectorConfig)


class GenerationStageConfig(BaseModel):
    canary_detector: DetectorConfig = Field(default_factory=DetectorConfig)
    language_detector: DetectorConfig = Field(default_factory=DetectorConfig)


class DefenseConfig(BaseModel):
    """Correspond to the detectors of the Deconvolute SDK."""

    ingestion: IngestionStageConfig = Field(default_factory=IngestionStageConfig)
    generation: GenerationStageConfig = Field(default_factory=GenerationStageConfig)
