from typing import Any, Literal

from pydantic import BaseModel, Field


class CanaryConfig(BaseModel):
    enabled: bool = Field(
        default=False, description="Whether canary defense is active."
    )
    settings: dict[str, Any] = Field(default_factory=dict)


class LanguageConfig(BaseModel):
    enabled: bool = Field(
        default=False, description="Whether language defense is active."
    )
    settings: dict[str, Any] = Field(default_factory=dict)


class SignatureConfig(BaseModel):
    enabled: bool = Field(
        default=False, description="Whether Signature defense is active."
    )
    settings: dict[str, Any] = Field(default_factory=dict)


class MLScannerConfig(BaseModel):
    enabled: bool = Field(
        default=False, description="Whether ML scanner defense is active."
    )
    settings: dict[str, Any] = Field(default_factory=dict)


class DefenseConfig(BaseModel):
    type: Literal["deconvolute", "none"] = Field(
        default="deconvolute", description="Defense provider."
    )
    strategy: Literal["layers", "guard"] = Field(
        default="layers",
        description=(
            "Integration strategy: 'layers' (manual) or 'guard' (orchestrator)."
        ),
    )
    required_version: str | None = Field(
        default=None, description="Min version required."
    )

    # Explicit Defense Layers
    canary: CanaryConfig | None = Field(default=None)
    language: LanguageConfig | None = Field(default=None)
    signature: SignatureConfig | None = Field(default=None)
    ml_scanner: MLScannerConfig | None = Field(default=None)
