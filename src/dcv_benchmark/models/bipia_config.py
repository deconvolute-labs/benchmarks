from typing import Literal

from pydantic import BaseModel, Field


class BipiaConfig(BaseModel):
    """
    Configuration for building the BIPIA dataset.
    """

    dataset_name: str = Field("bipia_v1", description="Name of the output dataset.")

    tasks: list[Literal["email", "code", "table", "qa"]] = Field(
        default=["email", "code", "table", "qa"],
        description="List of BIPIA tasks to include.",
    )

    injection_pos: Literal["start", "middle", "end"] = Field(
        "end", description="Where to inject the attack payload."
    )

    max_samples: int | None = Field(
        None, description="Limit number of samples per task."
    )
    seed: int = Field(42, description="Random seed.")
