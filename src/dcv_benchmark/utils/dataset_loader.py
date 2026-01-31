import json
from pathlib import Path

from dcv_benchmark.constants import BUILT_DATASETS_DIR
from dcv_benchmark.models.dataset import (
    BaseDataset,
    BipiaDataset,
    SquadDataset,
)


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.path = self._resolve_path(dataset_name)

    def _resolve_path(self, name: str) -> Path:
        """
        Resolves the dataset name to a file path.
        1. If it ends safely in .json, checks if it exists as a path.
        2. Else, assumes it's a directory name in BUILT_DATASETS_DIR/name/dataset.json.
        """
        # Convention-based check
        # workspace/datasets/built/{name}/dataset.json
        candidate = BUILT_DATASETS_DIR / name / "dataset.json"

        if candidate.exists():
            return candidate

        return Path(name)

    def load(self) -> BaseDataset:
        """
        Parses the dataset file and validates it against the schema.

        This method handles the deserialization of the JSON content into
        strict Pydantic models. It includes logic to auto-detect the dataset
        type (SQuAD vs BIPIA) based on metadata, defaulting to SQuAD/Canary
        for backward compatibility.

        Returns:
            BaseDataset: The validated dataset object.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
            ValueError: If the JSON is malformed or missing required fields.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        with open(self.path, encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in dataset file: {e}") from e

        # Determine type
        meta = raw_data.get("meta")
        if not meta:
            meta = {}
            raw_data["meta"] = meta

        dataset_type = meta.get("type")

        if dataset_type == "bipia":
            return BipiaDataset(**raw_data)
        elif dataset_type == "squad":
            return SquadDataset(**raw_data)

        # Fallback/Default
        # If no type, we now raise an error as strictly typed schemas are enforced.
        raise ValueError("Invalid dataset: Missing 'meta.type' field (squad/bipia).")
