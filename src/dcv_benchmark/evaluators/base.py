from abc import ABC, abstractmethod
from typing import Any

from dcv_benchmark.models.dataset import BenchmarkSample
from dcv_benchmark.models.evaluation import BaseEvaluationResult
from dcv_benchmark.models.responses import TargetResponse


class BaseEvaluator(ABC):
    """
    Abstract interface for judging a single interaction.

    Implementations should check `response.attack_detected` first.
    """

    @abstractmethod
    def evaluate(
        self, response: TargetResponse, sample: BenchmarkSample, **kwargs: Any
    ) -> BaseEvaluationResult:
        """
        Analyzes the target's response against the expected behavior defined
        in the sample.

        Args:
            response: The output from the System Under Test.
            sample: The ground truth input and expected behavior.
            **kwargs: Additional context needed for evaluation (e.g. security tokens).

        Returns:
            A result object containing pass/fail status, score, and reasoning.
        """
        pass
