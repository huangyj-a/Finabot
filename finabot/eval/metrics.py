"""Evaluation metrics per the report: pass rates, stability, severe failures.

Main metrics:
- Pass@1: fraction of trials that pass hard gates and quality threshold.
- Pass-all-N: fraction of tasks where all N trials pass (stability).
- Severe failure rate: vetoed trials / total trials, with 95% CI.
- Sub-metrics: claim support, calc recheck, future-leak, citation failure,
  conflict loss, refusal accuracy, latency/cost percentiles.
"""

from __future__ import annotations

import math
from typing import Any


def _internal_wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95% by default)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _internal_rule_of_three(n: int, alpha: float = 0.05) -> float:
    """Upper bound for zero failures: 1 - alpha**(1/n)."""
    if n <= 0:
        return 1.0
    return 1 - (1 - alpha) ** (1 / n)


def summarize_trials(trials: list[dict[str, Any]], quality_threshold: float = 75.0) -> dict[str, Any]:
    """Aggregate per-trial results into suite metrics.

    Parameters
    ----------
    trials
        Each trial dict: {task_id, pass_gates, quality, severe (bool),
        latency_ms, cost_cny, tool_errors, future_leak, ...}
    quality_threshold
        Minimum quality score for a trial to count as passed (report: dev 75).
    """
    n = len(trials)
    if n == 0:
        return {"n": 0, "note": "no trials"}

    pass_gates = [t for t in trials if t.get("pass_gates")]
    pass_quality = [t for t in pass_gates if t.get("quality", 0) >= quality_threshold]
    severe = [t for t in trials if t.get("severe")]
    future_leak = [t for t in trials if t.get("future_leak")]
    calc_fail = [t for t in trials if t.get("calc_pass_ratio", 1.0) < 1.0]
    tool_errors = [t for t in trials if (t.get("tool_errors") or 0) > 0]

    k_pass1 = len(pass_quality)
    pass1 = k_pass1 / n
    ci = _internal_wilson_ci(k_pass1, n)

    latencies = sorted(t["latency_ms"] for t in trials if t.get("latency_ms") is not None)
    costs = sorted(t["cost_cny"] for t in trials if t.get("cost_cny") is not None)

    return {
        "n": n,
        "pass_gates": len(pass_gates),
        "pass_quality": k_pass1,
        "Pass@1": round(pass1, 4),
        "Pass@1_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "severe_failure_rate": round(len(severe) / n, 4),
        "severe_failure_ci95": [round(v, 4) for v in _internal_wilson_ci(len(severe), n)],
        "future_leak_rate": round(len(future_leak) / n, 4),
        "calc_fail_rate": round(len(calc_fail) / n, 4),
        "tool_error_rate": round(len(tool_errors) / n, 4),
        "latency_ms": {
            "p50": _internal_percentile(latencies, 50),
            "p95": _internal_percentile(latencies, 95),
        },
        "cost_cny": {
            "p50": _internal_percentile(costs, 50),
            "p95": _internal_percentile(costs, 95),
        },
        "avg_cost_cny": round(sum(costs) / len(costs), 4) if costs else None,
        # rule of three: if severe failures == 0, upper bound is not zero
        "severe_zero_upper_bound": round(_internal_rule_of_three(n), 4) if len(severe) == 0 else None,
    }


def _internal_percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return round(sorted_values[lower], 2)
    frac = index - lower
    return round(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac, 2)


def pass_all_n(trials: list[dict[str, Any]], n_trials_per_task: int, quality_threshold: float = 75.0) -> dict[str, Any]:
    """Pass-all-N: fraction of tasks where every trial passes gates + quality."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_task.setdefault(trial.get("task_id", "?"), []).append(trial)

    passed_tasks = 0
    for task_trials in by_task.values():
        if len(task_trials) < n_trials_per_task:
            continue
        if all(
            t.get("pass_gates") and t.get("quality", 0) >= quality_threshold
            for t in task_trials[:n_trials_per_task]
        ):
            passed_tasks += 1

    total = sum(1 for task_trials in by_task.values() if len(task_trials) >= n_trials_per_task)
    return {
        "n_trials_per_task": n_trials_per_task,
        "pass_all_n": round(passed_tasks / total, 4) if total else None,
        "passed_tasks": passed_tasks,
        "total_tasks": total,
    }


def refusal_accuracy(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """拒绝准确性：该拒绝时的召回率 + 允许分析时不过度拒绝的精确率。

    Each trial should carry: refusal_expected (bool), refusal_given (bool),
    refusal_appropriate (bool).
    """
    n = len(trials)
    if n == 0:
        return {"n": 0}
    should_refuse = [t for t in trials if t.get("refusal_expected")]
    refused_when_needed = [t for t in should_refuse if t.get("refusal_given")]
    recall = len(refused_when_needed) / len(should_refuse) if should_refuse else None

    refused = [t for t in trials if t.get("refusal_given")]
    appropriate_refusals = [t for t in refused if t.get("refusal_appropriate")]
    precision = len(appropriate_refusals) / len(refused) if refused else None

    return {"n": n, "recall": round(recall, 4) if recall is not None else None, "precision": round(precision, 4) if precision is not None else None}