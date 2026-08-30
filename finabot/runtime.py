"""Background scheduling and heartbeat support for Finabot."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from finabot.agents.telemetry import snapshot_circuit_breaker, snapshot_llm_metrics, snapshot_subagent_metrics
from finabot.bus.queue import MessageBus


TaskCallback = Callable[[], Any | Awaitable[Any]]


def _internal_now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _internal_runtime_dir() -> Path:
    return Path(os.getenv("FINABOT_RUNTIME_DIR", "memory/runtime"))


@dataclass
class PeriodicTask:
    """A small async periodic task wrapper."""

    name: str
    interval_seconds: float
    callback: TaskCallback
    run_immediately: bool = False
    run_count: int = 0
    error_count: int = 0
    last_run_at: str | None = None
    last_error: str | None = None
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def run_once(self) -> None:
        try:
            result = self.callback()
            if inspect.isawaitable(result):
                await result
            self.run_count += 1
            self.last_run_at = _internal_now_iso()
            self.last_error = None
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.error_count += 1
            self.last_error = str(exc)

    async def loop(self) -> None:
        if self.run_immediately:
            await self.run_once()
        try:
            while True:
                await asyncio.sleep(self.interval_seconds)
                await self.run_once()
        except asyncio.CancelledError:
            return

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval_seconds": self.interval_seconds,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_run_at": self.last_run_at,
            "last_error": self.last_error,
        }


class RuntimeService:
    """Runs heartbeat and maintenance tasks beside the agent loop."""

    def __init__(
        self,
        agent: Any,
        bus: MessageBus,
        *,
        heartbeat_interval_seconds: float = 30,
        maintenance_interval_seconds: float = 300,
        diagnostic_interval_seconds: float = 0,
        runtime_dir: Path | None = None,
        diagnostic_thresholds: dict[str, Any] | None = None,
    ):
        self.agent = agent
        self.bus = bus
        self.runtime_dir = runtime_dir or _internal_runtime_dir()
        self.started_at = _internal_now_iso()
        self.heartbeat_count = 0
        self._tasks: list[PeriodicTask] = [
            PeriodicTask("heartbeat", heartbeat_interval_seconds, self.write_heartbeat, run_immediately=True),
            PeriodicTask("maintenance", maintenance_interval_seconds, self.run_maintenance),
        ]
        if diagnostic_interval_seconds > 0:
            # 诊断闭环：周期消费 snapshot()，检测异常并写入结构化诊断日志
            monitor = DiagnosticMonitor(self, log_dir=self.runtime_dir, thresholds=diagnostic_thresholds)
            self._tasks.append(
                PeriodicTask("diagnostic", diagnostic_interval_seconds, monitor.run, run_immediately=True)
            )

    @property
    def heartbeat_path(self) -> Path:
        return self.runtime_dir / "heartbeat.json"

    def add_task(self, task: PeriodicTask) -> None:
        self._tasks.append(task)

    async def start(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        for task in self._tasks:
            if task._task is None or task._task.done():
                task._task = asyncio.create_task(task.loop(), name=f"finabot:{task.name}")

    async def stop(self) -> None:
        running_tasks = [task._task for task in self._tasks if task._task and not task._task.done()]
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)

    async def run_maintenance(self) -> None:
        """清理超过 TTL 未活动的会话（checkpointer 线程 + 锁 + TTL 追踪）。"""

        cleanup = getattr(self.agent, "_cleanup_expired", None)
        if callable(cleanup):
            result = cleanup()
            if inspect.isawaitable(result):
                await result

    async def write_heartbeat(self) -> None:
        self.heartbeat_count += 1
        payload = self.snapshot()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        active_sessions = getattr(self.agent, "active_sessions", {})
        active_tasks = getattr(self.agent, "_tasks", set())
        return {
            "status": "running",
            "started_at": self.started_at,
            "heartbeat_at": _internal_now_iso(),
            "heartbeat_count": self.heartbeat_count,
            "bus": {
                "inbound_size": self.bus.inbound_size,
                "outbound_size": self.bus.outbound_size,
            },
            "agent": {
                "active_sessions": len(active_sessions) if isinstance(active_sessions, dict) else 0,
                "active_message_tasks": len(active_tasks) if isinstance(active_tasks, set) else 0,
                "session_locks": len(getattr(self.agent, "_session_locks", {})),
            },
            "llm": snapshot_llm_metrics(),
            "subagents": snapshot_subagent_metrics(),
            "circuit_breaker": snapshot_circuit_breaker(),
            "tasks": [task.snapshot() for task in self._tasks],
        }


_DEFAULT_DIAGNOSTIC_THRESHOLDS: dict[str, Any] = {
    "bus_backlog_warn": 10,
    "bus_backlog_error": 50,
    "llm_min_calls": 5,          # 样本量不足时不判失败率，避免偶发误报
    "llm_failure_rate_warn": 0.3,
    "llm_latency_warn_ms": 30000,
}


class DiagnosticMonitor:
    """诊断闭环：周期性消费 RuntimeService.snapshot()，检测异常并留痕。

    只做"判读 + 记录 + 送达"，不直接改动业务状态；告警送达支持可配置的
    webhook（FINABOT_ALERT_WEBHOOK_URL 或参数）与自定义 notifier 回调。检测项：
      - 心跳停摆：两次快照之间 heartbeat_count 未增长（运行时可能阻塞）；
      - 任务错误：任一周期任务 error_count 相比上次有增量（含错误文本）；
      - 总线积压：inbound/outbound 队列超过 warn/error 阈值；
      - LLM 异常：失败率或平均耗时超过阈值（样本量足够时）。
    每条异常写一行 JSON 到 `diagnostic.log`；error 级异常额外触发告警送达。
    """

    def __init__(
        self,
        runtime: RuntimeService,
        *,
        log_dir: Path | None = None,
        thresholds: dict[str, Any] | None = None,
        alert_webhook_url: str | None = None,
        notifier: Callable[[list[dict[str, str]]], Any | Awaitable[Any]] | None = None,
    ):
        self.runtime = runtime
        self.log_dir = Path(log_dir) if log_dir else (runtime.runtime_dir if runtime else _internal_runtime_dir())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = {**_DEFAULT_DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
        self.alert_webhook_url = alert_webhook_url or os.getenv("FINABOT_ALERT_WEBHOOK_URL") or ""
        self.notifier = notifier
        self._prev: dict[str, Any] | None = None
        self._seq = 0

    @property
    def log_path(self) -> Path:
        return self.log_dir / "diagnostic.log"

    async def run(self) -> None:
        """执行一次诊断：读取快照、对比上次、记录异常，并送达 error 级告警。"""
        self._seq += 1
        current = self.runtime.snapshot()
        issues = self._detect(current)
        self._prev = current
        errors: list[dict[str, str]] = []
        for issue in issues:
            self._append_log(issue)
            if issue["level"] == "error":
                errors.append(issue)
        if errors:
            await self._deliver(errors)

    async def _deliver(self, errors: list[dict[str, str]]) -> None:
        """把 error 级 issue 送达给 notifier 回调与 webhook（均尽力而为，失败不阻断）。"""
        if self.notifier is not None:
            try:
                result = self.notifier(errors)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # pragma: no cover - 通知失败不应影响主流程
                pass
        if self.alert_webhook_url:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._post_webhook_sync, errors),
                    timeout=5,
                )
            except Exception:  # pragma: no cover - webhook 不可达不阻断诊断
                pass

    def _post_webhook_sync(self, errors: list[dict[str, str]]) -> None:
        payload = json.dumps(
            {
                "type": "finabot_diagnostic_alert",
                "ts": _internal_now_iso(),
                "events": errors,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.alert_webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - 用户显式配置的 webhook
            response.read()

    def _detect(self, current: dict[str, Any]) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        prev = self._prev

        # 1) 心跳停摆（需要上一次快照做对比）
        if prev is not None:
            prev_count = prev.get("heartbeat_count", 0)
            curr_count = current.get("heartbeat_count", 0)
            if curr_count <= prev_count:
                issues.append({
                    "level": "warning",
                    "code": "heartbeat_stalled",
                    "message": f"heartbeat_count 未增长（{prev_count}→{curr_count}），运行时可能阻塞",
                })

        # 2) 周期任务错误增量
        prev_tasks = {t["name"]: t for t in prev.get("tasks", [])} if prev else {}
        for task in current.get("tasks", []):
            name = task["name"]
            prev_task = prev_tasks.get(name)
            if prev_task and task["error_count"] > prev_task["error_count"]:
                issues.append({
                    "level": "error",
                    "code": f"task_error:{name}",
                    "message": f"周期任务 {name} 新增错误（累计 {task['error_count']}）：{task.get('last_error') or '无详情'}",
                })

        # 3) 总线积压
        bus = current.get("bus", {})
        inbound = int(bus.get("inbound_size", 0) or 0)
        outbound = int(bus.get("outbound_size", 0) or 0)
        self._check_queue(issues, "inbound", inbound)
        self._check_queue(issues, "outbound", outbound)

        # 4) LLM 失败率 / 耗时
        llm = current.get("llm", {})
        calls = int(llm.get("calls", 0) or 0)
        if calls >= int(self.thresholds["llm_min_calls"]):
            failures = int(llm.get("failures", 0) or 0)
            rate = failures / calls if calls else 0.0
            if rate > float(self.thresholds["llm_failure_rate_warn"]):
                issues.append({
                    "level": "error",
                    "code": "llm_failure_rate",
                    "message": f"LLM 失败率 {rate:.0%}（{failures}/{calls}）超过阈值 {float(self.thresholds['llm_failure_rate_warn']):.0%}",
                })
            latency = float(llm.get("average_latency_ms", 0) or 0)
            if latency > float(self.thresholds["llm_latency_warn_ms"]):
                issues.append({
                    "level": "warning",
                    "code": "llm_high_latency",
                    "message": f"LLM 平均耗时 {latency:.0f}ms 超过阈值 {float(self.thresholds['llm_latency_warn_ms']):.0f}ms",
                })

        return issues

    def _check_queue(self, issues: list[dict[str, str]], name: str, size: int) -> None:
        error_threshold = int(self.thresholds["bus_backlog_error"])
        warn_threshold = int(self.thresholds["bus_backlog_warn"])
        if size > error_threshold:
            issues.append({"level": "error", "code": f"bus_backlog:{name}", "message": f"{name} 队列积压 {size} 条（> {error_threshold}）"})
        elif size > warn_threshold:
            issues.append({"level": "warning", "code": f"bus_backlog:{name}", "message": f"{name} 队列积压 {size} 条（> {warn_threshold}）"})

    def _append_log(self, issue: dict[str, str]) -> None:
        entry = {
            "seq": self._seq,
            "ts": _internal_now_iso(),
            "level": issue["level"],
            "code": issue["code"],
            "message": issue["message"],
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # pragma: no cover - 诊断日志写入失败不应拖垮主流程
            pass

