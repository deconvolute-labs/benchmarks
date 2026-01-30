import random
import re

from dcv_benchmark.data_factory.base import BaseInjector
from dcv_benchmark.models.data_factory import DataFactoryConfig
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


class AttackInjector(BaseInjector):
    """
    Injects malicious payloads using various adversarial strategies
    defined in the config.

    NOTE: This class is primarily used by the CLI 'data build' command
    (via SquadBuilder) and relies on 'DataFactoryConfig' which is not
    part of the runtime 'ExperimentConfig'. For runtime experiments,
    we assume datasets are pre-built.
    """

    def __init__(self, config: DataFactoryConfig):
        """
        Initializes the injector with the generation configuration.
        """
        self.config = config

    def inject(self, clean_text: str, payload: str, strategy: str) -> str:
        """
        Dispatches to the specific strategy method to inject payload into clean_text.

        Args:
            clean_text: The benign source text.
            payload: The malicious command to insert.
            strategy: The name of the strategy to use (overrides config if needed,
                      but usually passed from it).
        """
        strategies = {
            "naive": self._inject_naive,
            "prefix_injection": self._inject_prefix,
            "context_ignoring": self._inject_ignoring,
            "context_flooding": self._inject_flooding,
            "fake_system": self._inject_fake_system,
            "leet_speak": self._inject_leet,
        }

        if strategy not in strategies:
            logger.warning(f"Unknown strategy '{strategy}'. Falling back to 'naive'.")
            return self._inject_naive(clean_text, payload)

        return strategies[strategy](clean_text, payload)

    def _inject_naive(self, text: str, payload: str) -> str:
        """Strategy: Simply append the payload to the end."""
        return self._combine(text, payload, position="end")

    def _inject_prefix(self, text: str, payload: str) -> str:
        """Strategy: Prepend the payload to the beginning."""
        return self._combine(text, payload, position="start")

    def _inject_ignoring(self, text: str, payload: str) -> str:
        """Strategy: Wrap payload using the 'context_ignoring' template."""
        template = self.config.templates.get("context_ignoring")
        if not template:
            raise ValueError("No template provided!")
        wrapper = template.format(payload=payload)
        return self._combine(text, wrapper, position="end")

    def _inject_flooding(self, text: str, payload: str) -> str:
        """Strategy: Repeat the payload N times (defined in config)."""
        count = self.config.flooding_repetitions
        flooded_payload = " ".join([payload] * count)
        return self._combine(text, flooded_payload, position="end")

    def _inject_fake_system(self, text: str, payload: str) -> str:
        """Strategy: Wrap payload using the 'fake_system' template."""
        template = self.config.templates.get("fake_system")
        if not template:
            raise ValueError("No template provided!")
        wrapper = template.format(payload=payload)
        return self._combine(text, wrapper, position="end")

    def _inject_leet(self, text: str, payload: str) -> str:
        """Strategy: Obfuscate the payload with Leet Speak."""
        chars = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}
        leet_payload = "".join(chars.get(c.lower(), c) for c in payload)
        return self._combine(text, leet_payload, position="end")

    def _combine(self, text: str, payload: str, position: str) -> str:
        """
        Merges the benign text and the malicious payload, handling formatting and
        truncation.

        If `config.truncate_overflow` is True, the benign text is truncated to ensure
        the final length remains roughly equivalent to the original length.
        The payload is NEVER truncated, only the benign content is cut.

        Args:
            text: The original benign content.
            payload: The (potentially wrapped or flooded) attack string.
            position: 'start' (prepend) or 'end' (append).

        Returns:
            The combined string.
        """
        separator = "\n\n"

        # Case 1: No Truncation
        if not self.config.truncate_overflow:
            if position == "start":
                return f"{payload}{separator}{text}"
            return f"{text}{separator}{payload}"

        # Case 2: Truncation Required
        # We target the original length of the benign text
        target_len = len(text)
        payload_full_len = len(payload) + len(separator)

        # Edge case: Attack is bigger than the entire original chunk
        if payload_full_len >= target_len:
            return payload

        # Calculate allowed benign text length
        allowed_text_len = target_len - payload_full_len
        truncated_text = text[:allowed_text_len].strip() + "..."

        if position == "start":
            return f"{payload}{separator}{truncated_text}"
        return f"{truncated_text}{separator}{payload}"
