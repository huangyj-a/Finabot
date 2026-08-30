"""In-process operational metrics for LLM calls."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class LLMCallMetric:
    model: str
    started_at: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    retry: bool
    error_type: str | None = None


class LLMMetricsRegistry:
    """Keep aggregate counters and a bounded sample of recent LLM calls."""

    def __init__(self, recent_limit: int = 50):
        self._lock = threading.Lock()
        self._recent: deque[LLMCallMetric] = deque(maxlen=recent_limit)
        self._calls = 0
        self._failures = 0
        self._retries = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    def record(self, metric: LLMCallMetric) -> None:
        with self._lock:
            self._calls += 1
            self._failures += int(not metric.success)
            self._retries += int(metric.retry)
            self._prompt_tokens += metric.prompt_tokens
            self._completion_tokens += metric.completion_tokens
            self._total_tokens += metric.total_tokens
            self._total_latency_ms += metric.latency_ms
            self._recent.append(metric)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average_latency = self._total_latency_ms / self._calls if self._calls else 0.0
            recent = [asdict(item) for item in list(self._recent)[-10:]]
            return {
                "calls": self._calls,
                "failures": self._failures,
                "retries": self._retries,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._total_tokens,
                "average_latency_ms": round(average_latency, 2),
                "last_call_at": recent[-1]["started_at"] if recent else None,
                "recent": recent,
            }

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._calls = 0
            self._failures = 0
            self._retries = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0
            self._total_latency_ms = 0.0


LLM_METRICS = LLMMetricsRegistry()


@dataclass(frozen=True)
class SubagentMetric:
    """Per-sub-agent call metric (评估报告: 子代理维度可观测性)."""

    name: str
    latency_ms: float
    success: bool
    error_type: str | None = None


class SubagentMetricsRegistry:
    """Per-sub-agent counters: calls / failures / average latency."""

    def __init__(self, recent_limit: int = 50):
        self._lock = threading.Lock()
        self._per_name: dict[str, dict[str, Any]] = {}
        self._recent: deque[SubagentMetric] = deque(maxlen=recent_limit)

    def record(self, metric: SubagentMetric) -> None:
        with self._lock:
            entry = self._per_name.setdefault(
                metric.name,
                {"calls": 0, "failures": 0, "total_latency_ms": 0.0},
            )
            entry["calls"] += 1
            entry["failures"] += int(not metric.success)
            entry["total_latency_ms"] += metric.latency_ms
            self._recent.append(metric)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            per_name: dict[str, Any] = {}
            for name, entry in self._per_name.items():
                calls = entry["calls"]
                per_name[name] = {
                    "calls": calls,
                    "failures": entry["failures"],
                    "average_latency_ms": round(entry["total_latency_ms"] / calls, 2) if calls else 0.0,
                }
            return {
                "subagents": per_name,
                "recent": [
                    {"name": m.name, "latency_ms": m.latency_ms, "success": m.success, "error_type": m.error_type}
                    for m in list(self._recent)[-10:]
                ],
            }

    def reset(self) -> None:
        with self._lock:
            self._per_name.clear()
            self._recent.clear()


SUBAGENT_METRICS = SubagentMetricsRegistry()


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def snapshot_llm_metrics() -> dict[str, Any]:
    return LLM_METRICS.snapshot()


def snapshot_subagent_metrics() -> dict[str, Any]:
    return SUBAGENT_METRICS.snapshot()
