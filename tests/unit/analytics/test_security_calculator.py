import json

import pytest

from dcv_benchmark.analytics.calculators.security import SecurityMetricsCalculator


@pytest.fixture
def calculator():
    return SecurityMetricsCalculator()


@pytest.fixture
def traces_file(tmp_path):
    p = tmp_path / "traces.jsonl"
    return p


def create_trace(sample_type="benign", passed=True, strategy=None, latency=0.1):
    return json.dumps(
        {
            "sample_type": sample_type,
            "attack_strategy": strategy,
            "latency_seconds": latency,
            "evaluations": {"default": {"passed": passed}},
        }
    )


def test_calculator_file_not_found(calculator, tmp_path):
    with pytest.raises(FileNotFoundError):
        calculator.calculate(tmp_path / "non_existent.jsonl")


def test_calculator_empty_trace(calculator, traces_file):
    traces_file.touch()
    metrics = calculator.calculate(traces_file)

    assert metrics.global_metrics.total_samples == 0
    assert metrics.global_metrics.asr_score == 0.0
    assert metrics.global_metrics.pna_score == 1.0
    assert metrics.global_metrics.avg_latency_seconds == 0.0


def test_calculator_basic_benign_passed(calculator, traces_file):
    # Benign sample, defense passed (TN)
    content = create_trace(sample_type="benign", passed=True)
    traces_file.write_text(content, encoding="utf-8")

    metrics = calculator.calculate(traces_file)
    gm = metrics.global_metrics

    assert gm.total_samples == 1
    assert gm.tn == 1
    assert gm.fp == 0
    assert gm.pna_score == 1.0  # 1/1


def test_calculator_basic_benign_failed(calculator, traces_file):
    # Benign sample, defense failed (FP)
    content = create_trace(sample_type="benign", passed=False)
    traces_file.write_text(content, encoding="utf-8")

    metrics = calculator.calculate(traces_file)
    gm = metrics.global_metrics

    assert gm.total_samples == 1
    assert gm.tn == 0
    assert gm.fp == 1
    assert gm.pna_score == 0.0  # 0/1


def test_calculator_basic_attack_passed(calculator, traces_file):
    # Attack sample, defense passed (TP)
    content = create_trace(sample_type="attack", passed=True, strategy="jailbreak")
    traces_file.write_text(content, encoding="utf-8")

    metrics = calculator.calculate(traces_file)
    gm = metrics.global_metrics

    assert gm.total_samples == 1
    assert gm.tp == 1
    assert gm.fn == 0
    assert gm.asr_score == 0.0  # 0/1 failed


def test_calculator_basic_attack_failed(calculator, traces_file):
    # Attack sample, defense failed (FN -> Successful Attack)
    content = create_trace(sample_type="attack", passed=False, strategy="jailbreak")
    traces_file.write_text(content, encoding="utf-8")

    metrics = calculator.calculate(traces_file)
    gm = metrics.global_metrics

    assert gm.total_samples == 1
    assert gm.tp == 0
    assert gm.fn == 1
    assert gm.asr_score == 1.0  # 1/1 failed


def test_calculator_mixed_scenario(calculator, traces_file):
    lines = [
        create_trace(sample_type="benign", passed=True, latency=0.1),  # TN
        create_trace(sample_type="benign", passed=False, latency=0.2),  # FP
        create_trace(
            sample_type="attack", passed=True, strategy="strat1", latency=0.3
        ),  # TP
        create_trace(
            sample_type="attack", passed=False, strategy="strat1", latency=0.4
        ),  # FN
        create_trace(
            sample_type="attack", passed=False, strategy="strat2", latency=0.5
        ),  # FN
    ]
    traces_file.write_text("\n".join(lines), encoding="utf-8")

    metrics = calculator.calculate(traces_file)
    gm = metrics.global_metrics

    # Check totals
    assert gm.total_samples == 5
    assert gm.tn == 1
    assert gm.fp == 1
    assert gm.tp == 1
    assert gm.fn == 2

    # Check scores
    # ASR = FN / Total Attacks = 2 / 3 = 0.6667
    assert gm.asr_score == pytest.approx(0.6667, abs=0.0001)
    # PNA = TN / Total Benign = 1 / 2 = 0.5
    assert gm.pna_score == 0.5

    # Check Average Latency
    # (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 = 1.5 / 5 = 0.3
    assert gm.avg_latency_seconds == pytest.approx(0.3)

    # Check Strategy Breakdown
    strat1 = metrics.by_strategy["strat1"]
    assert strat1.samples == 2
    assert strat1.detected_count == 1
    assert strat1.missed_count == 1
    assert strat1.asr == 0.5

    strat2 = metrics.by_strategy["strat2"]
    assert strat2.samples == 1
    assert strat2.detected_count == 0
    assert strat2.missed_count == 1
    assert strat2.asr == 1.0
