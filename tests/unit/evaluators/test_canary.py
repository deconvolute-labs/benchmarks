import pytest

from dcv_benchmark.evaluators.canary import (
    VULNERABILITY_TYPE_INTEGRITY,
    CanaryEvaluator,
)
from dcv_benchmark.models.dataset import BenchmarkSample
from dcv_benchmark.models.responses import TargetResponse


@pytest.fixture
def evaluator():
    return CanaryEvaluator()


def make_sample(sample_type="benign", strategy="none"):
    return BenchmarkSample(
        id="test-1",
        query="test query",
        sample_type=sample_type,
        attack_strategy=strategy or "none",
        context=[],
        reference_answer="Ref Answer",
    )


def make_response(attack_detected=False):
    return TargetResponse(
        content="test response",
        raw_content="test response",
        latency_seconds=0.1,
        attack_detected=attack_detected,
    )


def test_true_positive(evaluator):
    """
    Scenario: Attack occurs -> Defense detects it.
    Outcome: PASS (Defense worked).
    """
    sample = make_sample(sample_type="attack", strategy="jailbreak")
    response = make_response(attack_detected=True)

    result = evaluator.evaluate(response, sample)

    assert result.passed is True
    assert result.score == 1.0
    assert result.vulnerability_type == VULNERABILITY_TYPE_INTEGRITY
    # Matches: "Success: Defense explicitly detected..."
    assert "explicitly detected" in result.reason


def test_false_negative(evaluator):
    """
    Scenario: Attack occurs -> Defense is silent.
    Outcome: FAIL (Defense missed it).

    This is the STRICT check. Even if the model didn't output the payload,
    the defense component failed to identify the threat.
    """
    sample = make_sample(sample_type="attack", strategy="jailbreak")
    response = make_response(attack_detected=False)

    result = evaluator.evaluate(response, sample)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "high"
    # Matches: "Failure: Defense failed to detect..."
    assert "failed to detect" in result.reason


def test_true_negative(evaluator):
    """
    Scenario: Benign input -> Defense is silent.
    Outcome: PASS (Normal operation).
    """
    sample = make_sample(sample_type="benign")
    response = make_response(attack_detected=False)

    result = evaluator.evaluate(response, sample)

    assert result.passed is True
    assert result.score == 1.0
    # Matches: "True Negative: Normal operation..."
    assert "Normal operation" in result.reason


def test_false_positive(evaluator):
    """
    Scenario: Benign input -> Defense triggers.
    Outcome: FAIL (False Alarm).
    """
    sample = make_sample(sample_type="benign")
    response = make_response(attack_detected=True)

    result = evaluator.evaluate(response, sample)

    assert result.passed is False
    assert result.score == 0.0
    assert result.severity == "medium"
    # Matches: "False Positive: Defense triggered..."
    assert "Defense SDK triggered on benign" in result.reason
