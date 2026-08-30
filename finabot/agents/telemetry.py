"""In-process operational metrics for LLM calls."""

from __future__ import annotations

import threading
import time
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


class LLMCircuitBreaker:
    """连续 LLM 失败熔断：达到阈值后冷却期内直接拒绝新调用，避免反复打爆下游。

    评估报告"预算"与 loop 护栏：端点不可用时不应让每个请求都耗尽重试预算。
    冷却结束后自动半开（重置连续失败计数），下次成功即彻底关闭。
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_until = 0.0
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_until:
                if time.monotonic() < self._opened_until:
                    return True
                # 冷却结束：半开（重置计数，等待下一次调用验证）
                self._opened_until = 0.0
                self._consecutive_failures = 0
            return False

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_until = time.monotonic() + self.cooldown_seconds

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            open_state = bool(self._opened_until and now < self._opened_until)
            return {
                "open": open_state,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
            }


LLM_CIRCUIT_BREAKER = LLMCircuitBreaker()


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def snapshot_llm_metrics() -> dict[str, Any]:
    return LLM_METRICS.snapshot()


def snapshot_subagent_metrics() -> dict[str, Any]:
    return SUBAGENT_METRICS.snapshot()


def snapshot_circuit_breaker() -> dict[str, Any]:
    return LLM_CIRCUIT_BREAKER.snapshot()
