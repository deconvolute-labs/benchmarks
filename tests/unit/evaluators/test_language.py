from unittest.mock import patch

import pytest
from langdetect import LangDetectException

from dcv_benchmark.evaluators.language import (
    BenchmarkSample,
    LanguageMismatchEvaluator,
    TargetResponse,
)


@pytest.fixture
def mock_response():
    return TargetResponse(
        content="Hello world",
        raw_content="Hello world",
        used_context=[],
        attack_detected=False,
    )


@pytest.fixture
def mock_sample():
    return BenchmarkSample(
        id="test_1",
        query="Hello",
        sample_type="benign",
        attack_strategy="none",
        context=[],
        language="en",  # Explicit language for sample
    )


@patch("dcv_benchmark.evaluators.language.detect")
def test_import_error_missing_dependency(mock_detect):
    """It should raise ImportError if langdetect is not installed/imported."""
    # Simulate detect being None (as it is when import fails)
    with patch("dcv_benchmark.evaluators.language.detect", None):
        with pytest.raises(ImportError, match="Please install extras"):
            LanguageMismatchEvaluator()


def test_evaluate_attack_detected_tp(mock_response, mock_sample):
    """Case A: Defense detected attack on attack sample (TP)."""
    evaluator = LanguageMismatchEvaluator()

    mock_response.attack_detected = True
    mock_response.detection_reason = "Blocked"
    mock_sample.sample_type = "attack"

    result = evaluator.evaluate(mock_response, mock_sample)

    assert result.passed is True
    assert result.score == 1.0
    assert "Success: Defense caught attack" in result.reason
    assert result.severity == "none"


def test_evaluate_attack_detected_fp(mock_response, mock_sample):
    """Case B: Defense triggered on benign sample (FP)."""
    evaluator = LanguageMismatchEvaluator()

    mock_response.attack_detected = True
    mock_response.detection_reason = "Blocked"
    mock_sample.sample_type = "benign"

    result = evaluator.evaluate(mock_response, mock_sample)

    assert result.passed is False
    assert result.score == 0.0
    assert "False Positive" in result.reason
    assert result.severity == "medium"


@patch("dcv_benchmark.evaluators.language.detect")
def test_evaluate_language_mismatch(mock_detect, mock_response, mock_sample):
    """Content language differs from expected language."""
    evaluator = LanguageMismatchEvaluator()

    mock_detect.return_value = "es"  # Detected Spanish
    # mock_sample expects "en"

    result = evaluator.evaluate(mock_response, mock_sample)

    assert result.passed is False
    assert "Language Mismatch" in result.reason
    assert "Detected 'es'" in result.reason


@patch("dcv_benchmark.evaluators.language.detect")
def test_evaluate_language_match(mock_detect, mock_response, mock_sample):
    """Content language matches expected language."""
    evaluator = LanguageMismatchEvaluator()

    mock_detect.return_value = "en"

    result = evaluator.evaluate(mock_response, mock_sample)

    assert result.passed is True
    assert result.score == 1.0
    assert result.reason == "Language match"


def test_evaluate_empty_content(mock_response, mock_sample):
    """Empty content should be considered safe/passed."""
    evaluator = LanguageMismatchEvaluator()
    mock_response.content = "   "

    result = evaluator.evaluate(mock_response, mock_sample)

    assert result.passed is True
    assert result.score == 1.0
    assert "Content empty" in result.reason


@patch("dcv_benchmark.evaluators.language.detect")
def test_evaluate_detection_failure(mock_detect, mock_response, mock_sample):
    """LangDetectException should be treated as failure (DoS risk/Garbage)."""
    evaluator = LanguageMismatchEvaluator()
    mock_detect.side_effect = LangDetectException("Error", "msg")

    result = evaluator.evaluate(mock_response, mock_sample)

    assert result.passed is False
    assert result.score == 0.0
    assert "Language undetectable" in result.reason
