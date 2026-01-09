import json
from pathlib import Path

from dcv_benchmark.models.dataset import Dataset


class DatasetLoader:
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)

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
