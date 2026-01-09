from typing import Any, Literal

from pydantic import BaseModel, Field


class CorpusInfo(BaseModel):
    """Metadata about the source documents used for this dataset."""

    source_files: list[str]
    pre_chunked_file: str | None = None
    ingestion_params: dict[str, int | str] = Field(default_factory=dict)


class ContextChunk(BaseModel):
    """A single retrieved text chunk."""

    id: str
    content: str
    is_malicious: bool = False
    # Logic specific to this chunk
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSample(BaseModel):
    """A single test case (Query + Expected Outcome)."""

    id: str
    query: str

    sample_type: Literal["attack", "benign"]

    # The specific attack stratey (e.g. "leet_speak", "context_flooding")
    attack_strategy: str = "none"

    # The expected 'correct' answer (mostly for benign utility checks)
    reference_answer: str | None = None

    context: list[ContextChunk]


class DatasetMeta(BaseModel):
    """Top-level dataset metadata."""

    name: str
    version: str
    description: str
    author: str
    corpus_info: CorpusInfo | None = None


class Dataset(BaseModel):
    """The full dataset file structure."""

    meta: DatasetMeta
    samples: list[BenchmarkSample]
