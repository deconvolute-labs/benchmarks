import random
import re

from dcv_benchmark.data_factory.base import BaseInjector
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


class BipiaInjector(BaseInjector):
    """
    Injector implementation for BIPIA strategies (start, middle, end).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def inject(self, clean_text: str, payload: str, strategy: str) -> str:
        """
        Injects payload based on BIPIA strategy ('start', 'middle', 'end').
        """
        if strategy == "start":
            return self._insert_start(clean_text, payload)
        elif strategy == "middle":
            return self._insert_middle(clean_text, payload)
        elif strategy == "end":
            return self._insert_end(clean_text, payload)
        else:
            logger.warning(f"Unknown BIPIA strategy '{strategy}'. Defaulting to 'end'.")
            return self._insert_end(clean_text, payload)

    def _split_sentences(self, text: str) -> list[str]:
        """
        Splits text into sentences using regex to avoid NLTK dependency.
        Approximates the behavior of PunktSentenceTokenizer.
        """
        # Split on period/question/exclamation followed by space or end of string.
        # We keep the delimiter with the sentence.
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    def _insert_end(self, context: str, attack: str) -> str:
        """Ported from: https://github.com/microsoft/BIPIA/blob/main/bipia/data/utils.py"""
        return "\n".join([context, attack])

    def _insert_start(self, context: str, attack: str) -> str:
        """Ported from: https://github.com/microsoft/BIPIA/blob/main/bipia/data/utils.py"""
        return "\n".join([attack, context])

    def _insert_middle(self, context: str, attack: str) -> str:
        """
        Adapted from: https://github.com/microsoft/BIPIA/blob/main/bipia/data/utils.py
        Uses class-level RNG seeded in __init__.
        """
        sentences = self._split_sentences(context)
        if not sentences:
            return self._insert_end(context, attack)

        # Pick a random split point (avoiding index 0 to truly be 'middle' if possible)
        if len(sentences) > 1:
            split_idx = random.randint(1, len(sentences) - 1)  # noqa
        else:
            split_idx = 0

        pre = " ".join(sentences[:split_idx])
        post = " ".join(sentences[split_idx:])

        return "\n".join([pre, attack, post])
