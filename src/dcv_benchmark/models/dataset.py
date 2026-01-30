from typing import Any, Literal

from pydantic import BaseModel, Field


class CorpusInfo(BaseModel):
    """Metadata about the source documents used for this dataset."""

    source_files: list[str] = Field(..., description="List of source file paths.")
    pre_chunked_file: str | None = Field(
        default=None, description="Path to pre-chunked file if exists."
    )
    ingestion_params: dict[str, int | str] = Field(default_factory=dict)


class AttackInfo(BaseModel):
    """Metadata about the attack logic used to generate this dataset."""

    strategy: str = Field(..., description="Attack strategy used.")
    rate: float = Field(..., description="Proportion of attack samples.")
    payload: str = Field(..., description="Injection payload.")
    configuration: dict[str, Any] = Field(default_factory=dict)


class ContextChunk(BaseModel):
    """A single retrieved text chunk."""

    id: str = Field(..., description="Chunk ID.")
    content: str = Field(..., description="Text content.")
    is_malicious: bool = Field(
        default=False, description="Whether this chunk contains a malicious payload."
    )
    # Logic specific to this chunk
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSample(BaseModel):
    """A single test case (Query + Expected Outcome)."""

    id: str = Field(..., description="Sample ID.")
    query: str = Field(..., description="User query.")

    sample_type: Literal["attack", "benign"] = Field(..., description="Type of sample.")

    # The specific attack stratey (e.g. "leet_speak", "context_flooding")
    attack_strategy: str = Field(
        default="none", description="Specific attack strategy."
    )

    language: str | None = Field(
        default=None,
        description=(
            "Expected ISO language code (e.g. 'en'). Overrides global defaults."
        ),
    )

    # The expected 'correct' answer (mostly for benign utility checks)
    reference_answer: str | None = Field(default=None, description="Expected answer.")

    context: list[ContextChunk] = Field(
        default_factory=list, description="Retrieved context chunks."
    )


class DatasetMeta(BaseModel):
    """Top-level dataset metadata."""

    name: str = Field(..., description="Dataset name.")
    type: Literal["squad", "bipia"] = Field(..., description="Dataset type.")
    version: str = Field(..., description="Dataset version.")
    description: str = Field(..., description="Dataset description.")
    author: str = Field(..., description="Dataset author.")
    corpus_info: CorpusInfo | None = Field(default=None, description="Corpus metadata.")
    attack_info: AttackInfo | None = Field(default=None, description="Attack metadata.")


class BaseDataset(BaseModel):
    """The full dataset file structure - Base Class."""

    meta: DatasetMeta = Field(..., description="Dataset metadata.")
    samples: list[BenchmarkSample] = Field(..., description="List of samples.")


class SquadDataset(BaseDataset):
    """Dataset class for SQuAD/Canary style datasets."""

    pass


class BipiaDataset(BaseDataset):
    """Dataset class for BIPIA style datasets."""

    pass


# For backward compatibility
Dataset = BaseDataset
