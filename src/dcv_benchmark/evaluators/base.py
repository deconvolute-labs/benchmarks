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
        Analyzes a single interaction to determine pass/fail status.

        Implementations should compare the `response` against the `sample` ground
        truth. Security evaluators often check `response.attack_detected` first
        before analyzing the content.

        Args:
            response (TargetResponse): The output object from the System Under Test,
                containing the generated text and defense metadata.
            sample (BenchmarkSample): The ground truth object containing the original
                query, expected answer, and attack strategy details.
            **kwargs (Any): Additional context required for specific evaluators.
                For example, a Canary evaluator might need the `canary_token`
                extracted from the prompt.

        Returns:
            BaseEvaluationResult: The verdict of the evaluation, including a boolean
            `passed` flag, a score (0.0-1.0), and a reasoning string.
        """
        pass
