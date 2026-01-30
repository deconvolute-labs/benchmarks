from abc import ABC, abstractmethod
from typing import Any

from dcv_benchmark.models.data_factory import RawSample


class BaseCorpusLoader(ABC):
    """
    Abstract Base Class for Corpus Loaders.

    A Loader is responsible for reading a specific file format (JSON, CSV, Parquet)
    and normalizing it into a list of `RawSample` objects.
    """

    @abstractmethod
    def load(self, file_path: str) -> list[RawSample]:
        """
        Loads the file from disk and normalizes it.

        Args:
            file_path: Absolute or relative path to the corpus file.

        Returns:
            A list of validated RawSample objects.

        Raises:
            FileNotFoundError: If the file_path does not exist.
            ValueError: If the file content cannot be parsed.
        """
        pass


class BaseInjector(ABC):
    """
    Abstract Base Class for Attack Injectors ('Weaponizers').

    An Injector is responsible for taking benign text and embedding a malicious
    payload according to a specific strategy.
    """

    @abstractmethod
    def inject(self, clean_text: str, payload: str, strategy: str) -> str:
        """
        Injects the payload into the clean_text.

        Args:
            clean_text: The original, benign content chunk.
            payload: The malicious string to insert.
            strategy: The name of the strategy
                      (allows a single Injector to handle multiple variants).

        Returns:
            The modified text string containing the attack.
        """
        pass


class BaseDatasetBuilder(ABC):
    """
    Abstract Base Class for Dataset Builders.
    """

    @abstractmethod
    def build(self, **kwargs: Any) -> Any:
        # TODO: The return type should ideally be `Dataset` but we need to
        # TODO: resolve circular imports
        # or use ForwardRef / 'Dataset'. For now `Any` is permissive.
        """
        Builds and returns the dataset.
        """
        pass
