"""Tests for eval metrics (Pass@1, Pass-all-N, severe failure CI)."""

from finabot.eval.metrics import pass_all_n, refusal_accuracy, summarize_trials


def _trial(task_id, pass_gates=True, quality=80.0, severe=False, latency=1000.0, cost=0.5,
           future_leak=False, calc_ratio=1.0, tool_errors=0,
           refusal_expected=False, refusal_given=False, refusal_appropriate=True):
    return {
        "task_id": task_id,
        "pass_gates": pass_gates,
        "quality": quality,
        "severe": severe,
        "latency_ms": latency,
        "cost_cny": cost,
        "future_leak": future_leak,
        "calc_pass_ratio": calc_ratio,
        "tool_errors": tool_errors,
        "refusal_expected": refusal_expected,
        "refusal_given": refusal_given,
        "refusal_appropriate": refusal_appropriate,
    }


def test_summarize_pass_rate_and_severe_ci():
    trials = [_trial("t1") for _ in range(4)] + [_trial("t1", pass_gates=False, severe=True)]
    summary = summarize_trials(trials, quality_threshold=75.0)
    assert summary["n"] == 5
    assert summary["Pass@1"] == 0.8
    assert summary["severe_failure_rate"] == 0.2
    assert summary["severe_failure_ci95"][0] <= 0.2 <= summary["severe_failure_ci95"][1]
    assert summary["latency_ms"]["p50"] == 1000.0


def test_rule_of_three_upper_bound_nonzero():
    trials = [_trial("t1") for _ in range(5)]
    summary = summarize_trials(trials, quality_threshold=75.0)
    # 零严重失败时上界不为 0（rule of three）
    assert summary["severe_zero_upper_bound"] is not None
    assert summary["severe_zero_upper_bound"] > 0


def test_pass_all_n():
    trials = [_trial("t1") for _ in range(3)] + [_trial("t2") for _ in range(3)]
    trials[5] = _trial("t2", pass_gates=False)
    result = pass_all_n(trials, n_trials_per_task=3, quality_threshold=75.0)
    assert result["pass_all_n"] == 0.5


def test_refusal_accuracy():
    trials = [
        _trial("t1", refusal_expected=True, refusal_given=True),
        _trial("t2", refusal_expected=True, refusal_given=False),
        _trial("t3", refusal_expected=False, refusal_given=True, refusal_appropriate=False),
        _trial("t4", refusal_expected=False, refusal_given=False),
    ]
    result = refusal_accuracy(trials)
    assert result["recall"] == 0.5
    assert result["precision"] == 0.5