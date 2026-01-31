from unittest.mock import MagicMock

import pytest

from dcv_benchmark.components.llms import BaseLLM
from dcv_benchmark.evaluators.bipia import BipiaDefenseEvaluator


class MockLLM(BaseLLM):
    def __init__(self):
        pass

    def generate(self, system_message: str, user_message: str) -> str | None:
        return "NO"  # Default safe response


@pytest.fixture
def mock_judge_llm():
    llm = MockLLM()
    llm.generate = MagicMock(return_value="NO")
    return llm


def test_bipia_init_defaults():
    evaluator = BipiaDefenseEvaluator()
    assert evaluator.judge_llm is None


def test_bipia_init_with_judge(mock_judge_llm):
    evaluator = BipiaDefenseEvaluator(judge_llm=mock_judge_llm)
    assert evaluator.judge_llm == mock_judge_llm


def test_eval_llm_judge_safe(mock_judge_llm):
    """Test that a 'NO' response from the judge results in a safe evaluation."""
    evaluator = BipiaDefenseEvaluator(judge_llm=mock_judge_llm)

    # Mock behavior: Judge says NO (not an attack)
    mock_judge_llm.generate.return_value = "NO"

    result = evaluator._eval_llm_judge("some text", "Task Automation", 0)

    assert result.passed is True
    assert result.score == 1.0
    mock_judge_llm.generate.assert_called_once()
    assert "Judge: NO" in result.reason


def test_eval_llm_judge_unsafe(mock_judge_llm):
    """Test that a 'YES' response from the judge results in an unsafe evaluation."""
    evaluator = BipiaDefenseEvaluator(judge_llm=mock_judge_llm)

    # Mock behavior: Judge says YES (is an attack)
    mock_judge_llm.generate.return_value = "YES"

    result = evaluator._eval_llm_judge("some text", "Task Automation", 0)

    assert result.passed is False
    assert result.score == 0.0
    mock_judge_llm.generate.assert_called_once()


def test_eval_llm_judge_missing_config():
    """Test that RuntimeError is raised when judge is needed but missing."""
    evaluator = BipiaDefenseEvaluator(judge_llm=None)

    with pytest.raises(RuntimeError, match="Evaluation requires a Judge LLM"):
        evaluator._eval_llm_judge("some text", "Task Automation", 0)
