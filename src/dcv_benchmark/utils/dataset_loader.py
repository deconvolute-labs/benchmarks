import json
from pathlib import Path

from dcv_benchmark.constants import BUILT_DATASETS_DIR
from dcv_benchmark.models.dataset import Dataset


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.path = self._resolve_path(dataset_name)

    def _resolve_path(self, name: str) -> Path:
        """
        Resolves the dataset name to a file path.
        1. If it ends safely in .json, checks if it exists as a path.
        2. Else, assumes it's a directory name in BUILT_DATASETS_DIR/name/dataset.json.
        """
        # 1. Direct path check (backward compatibility)
        if name.endswith(".json"):
            direct_path = Path(name)
            if direct_path.exists():
                return direct_path

        # 2. Convention-based check
        # workspace/datasets/built/{name}/dataset.json
        candidate = BUILT_DATASETS_DIR / name / "dataset.json"

        # If the candidate doesn't exist, we fallback to just returning it
        # so valid file-not-found error raises in .load() or we can check here.
        # But let's check here to return the most helpful path for the error message.
        if candidate.exists():
            return candidate

        return candidate if not name.endswith(".json") else Path(name)

    def load(self) -> Dataset:
        """
        Reads the JSON file, validates it, and returns a Pydantic Dataset object.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValidationError: If the JSON structure doesn't match the schema.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        with open(self.path, encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in dataset file: {e}") from e

        return Dataset(**raw_data)
