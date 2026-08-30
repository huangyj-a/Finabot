"""Tests for background runtime scheduling and heartbeat."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from finabot.bus.queue import MessageBus
from finabot.bus.events import InboundMessage
from finabot.runtime import PeriodicTask, RuntimeService


def _runtime_test_dir() -> Path:
    path = Path("memory") / f"runtime_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class DummyAgent:
    def __init__(self):
        self._thread_last_used = {}
        self._tasks = set()
        self._session_locks = {}

    async def _cleanup_expired(self):
        now = datetime.now()
        expired_keys = [
            key for key, last in self._thread_last_used.items()
            if now - last > timedelta(minutes=60)
        ]
        for key in expired_keys:
            self._thread_last_used.pop(key, None)
            self._session_locks.pop(key, None)
        return expired_keys

    @property
    def active_sessions(self):
        return dict(self._thread_last_used)


@pytest.mark.anyio
async def test_periodic_task_runs_callback_repeatedly():
    calls = []

    async def callback():
        calls.append("ok")

    task = PeriodicTask("test", 0.01, callback, run_immediately=True)
    runner = asyncio.create_task(task.loop())
    await asyncio.sleep(0.035)
    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)

    assert len(calls) >= 2
    assert task.run_count >= 2
    assert task.last_error is None


@pytest.mark.anyio
async def test_runtime_writes_heartbeat_file():
    runtime_dir = _runtime_test_dir()
    try:
        agent = DummyAgent()
        bus = MessageBus()
        runtime = RuntimeService(agent, bus, heartbeat_interval_seconds=0.01, runtime_dir=runtime_dir)

        await runtime.write_heartbeat()

        payload = json.loads(runtime.heartbeat_path.read_text(encoding="utf-8"))

        assert payload["status"] == "running"
        assert payload["heartbeat_count"] == 1
        assert payload["bus"] == {"inbound_size": 0, "outbound_size": 0}
        assert payload["agent"]["active_sessions"] == 0
        assert payload["agent"]["session_locks"] == 0
        assert payload["llm"]["calls"] >= 0
        assert "total_tokens" in payload["llm"]
        assert "average_latency_ms" in payload["llm"]
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_runtime_maintenance_cleans_expired_threads():
    runtime_dir = _runtime_test_dir()
    try:
        agent = DummyAgent()
        key = "cli:expired"
        expired = datetime.now() - timedelta(hours=2)
        agent._thread_last_used[key] = expired
        agent._session_locks[key] = asyncio.Lock()
        runtime = RuntimeService(agent, MessageBus(), runtime_dir=runtime_dir)

        await runtime.run_maintenance()

        assert key not in agent._thread_last_used
        assert key not in agent._session_locks
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_periodic_task_error_does_not_kill_loop():
    """回调抛错不应终止定时循环（7×24 自愈前提：单个任务失败不拖垮整个进程）。"""
    calls = {"ok": 0, "fail": 0}

    def callback():
        if calls["fail"] < 1:
            calls["fail"] += 1
            raise RuntimeError("boom once")
        calls["ok"] += 1
        return "fine"

    task = PeriodicTask("probe", 0.01, callback, run_immediately=True)
    runner = asyncio.create_task(task.loop())
    await asyncio.sleep(0.035)
    runner.cancel()
    await asyncio.gather(runner, return_exceptions=True)

    # 失败被计入 error_count（可诊断），且循环继续运转（后续成功执行，自愈）
    assert task.error_count == 1
    assert calls["ok"] >= 1
    assert task.run_count >= 1
    # 成功后 last_error 被清空，避免错误信号残留造成误报
    assert task.last_error is None


@pytest.mark.anyio
async def test_persistent_failure_is_visible_for_diagnosis():
    """持续失败的周期任务应把具体错误暴露出来，供日志/告警定位。"""

    async def always_fail():
        raise RuntimeError("api_timeout: akshare")

    task = PeriodicTask("probe", 1.0, always_fail, run_immediately=True)
    await task.run_once()

    assert task.error_count == 1
    assert task.last_error is not None
    assert "api_timeout" in task.last_error


@pytest.mark.anyio
async def test_heartbeat_carries_fault_diagnosis_signals():
    """心跳应包含足以定位故障的观测信号：bus 积压、活动会话/任务/锁、LLM 失败与耗时、任务自身错误。"""
    runtime_dir = _runtime_test_dir()
    try:
        agent = DummyAgent()
        bus = MessageBus()
        await bus.publish_inbound(InboundMessage(
            channel="cli", sender_id="t", chat_id="c1", content="hi"
        ))
        agent._thread_last_used["cli:c1"] = datetime.now()
        agent._session_locks["cli:c1"] = asyncio.Lock()
        agent._tasks.add(asyncio.current_task())

        runtime = RuntimeService(agent, bus, runtime_dir=runtime_dir)
        await runtime.write_heartbeat()

        payload = json.loads(runtime.heartbeat_path.read_text(encoding="utf-8"))
        # 会话/任务/锁：异常堆积时一眼可见
        assert payload["agent"]["active_sessions"] == 1
        assert payload["agent"]["session_locks"] == 1
        assert payload["agent"]["active_message_tasks"] == 1
        # 总线积压：消息卡住时的关键信号
        assert payload["bus"]["inbound_size"] == 1
        # LLM 观测：调用量、失败数、重试、耗时、最近一次调用时间
        llm = payload["llm"]
        for key in ("calls", "failures", "retries", "total_tokens", "average_latency_ms", "last_call_at"):
            assert key in llm
        # 每个后台任务都暴露 run/error 计数，失败可被定位到具体任务
        task_snapshots = {t["name"]: t for t in payload["tasks"]}
        assert set(task_snapshots) >= {"heartbeat", "maintenance"}
        for t in task_snapshots.values():
            assert "run_count" in t and "error_count" in t and "last_error" in t
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_runtime_start_stop_lifecycle():
    runtime_dir = _runtime_test_dir()
    try:
        agent = DummyAgent()
        runtime = RuntimeService(agent, MessageBus(), heartbeat_interval_seconds=0.01, maintenance_interval_seconds=60, runtime_dir=runtime_dir)

        await runtime.start()
        # 立即运行 + 周期运行：心跳任务至少已执行过一次
        await asyncio.sleep(0.03)
        await runtime.stop()

        # 心跳文件已写入，且所有后台任务已取消
        assert runtime.heartbeat_path.exists()
        assert all(task._task is None or task._task.done() for task in runtime._tasks)
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


class _FakeRuntimeSnapshots:
    """按顺序返回预置快照的假 runtime，供 DiagnosticMonitor 单测。"""

    def __init__(self, snapshots):
        self._snapshots = list(snapshots)
        self._index = 0

    def snapshot(self):
        snapshot = self._snapshots[self._index]
        self._index = min(self._index + 1, len(self._snapshots) - 1)
        return snapshot


def _diag_snapshot(**overrides):
    base = {
        "heartbeat_count": 1,
        "bus": {"inbound_size": 0, "outbound_size": 0},
        "tasks": [],
        "llm": {"calls": 0, "failures": 0, "average_latency_ms": 0},
    }
    base.update(overrides)
    return base


def test_diagnostic_detects_bus_backlog(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    fake = _FakeRuntimeSnapshots([
        _diag_snapshot(bus={"inbound_size": 60, "outbound_size": 12}),
    ])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path)
    issues = monitor._detect(fake.snapshot())

    codes = [issue["code"] for issue in issues]
    assert "bus_backlog:inbound" in codes  # 60 > 50 → error
    assert "bus_backlog:outbound" in codes  # 12 > 10 → warn
    inbound_issue = next(issue for issue in issues if issue["code"] == "bus_backlog:inbound")
    assert inbound_issue["level"] == "error"


def test_diagnostic_detects_task_error_increment(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    prev = _diag_snapshot(tasks=[{"name": "heartbeat", "error_count": 0, "last_error": None}])
    current = _diag_snapshot(
        heartbeat_count=2,
        tasks=[{"name": "heartbeat", "error_count": 3, "last_error": "api_timeout: akshare"}],
    )
    fake = _FakeRuntimeSnapshots([prev, current])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path)
    monitor._prev = prev
    issues = monitor._detect(current)

    task_issue = next(issue for issue in issues if issue["code"] == "task_error:heartbeat")
    assert task_issue["level"] == "error"
    assert "api_timeout" in task_issue["message"]


def test_diagnostic_detects_heartbeat_stall(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    snapshot = _diag_snapshot(heartbeat_count=5)
    fake = _FakeRuntimeSnapshots([snapshot])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path)
    monitor._prev = snapshot
    issues = monitor._detect(snapshot)

    assert any(issue["code"] == "heartbeat_stalled" for issue in issues)


def test_diagnostic_detects_llm_failure_rate(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    snapshot = _diag_snapshot(llm={"calls": 10, "failures": 6, "average_latency_ms": 800})
    fake = _FakeRuntimeSnapshots([snapshot])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path)
    issues = monitor._detect(snapshot)

    assert any(issue["code"] == "llm_failure_rate" for issue in issues)
    assert not any(issue["code"] == "llm_high_latency" for issue in issues)


@pytest.mark.anyio
async def test_diagnostic_run_writes_jsonl_log(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    fake = _FakeRuntimeSnapshots([
        _diag_snapshot(bus={"inbound_size": 60, "outbound_size": 0}),
        _diag_snapshot(bus={"inbound_size": 60, "outbound_size": 0}),
    ])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path)
    await monitor.run()

    assert monitor.log_path.exists()
    lines = monitor.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    entry = json.loads(lines[0])
    assert entry["level"] == "error"
    assert entry["code"] == "bus_backlog:inbound"
    assert "seq" in entry and "ts" in entry


@pytest.mark.anyio
async def test_runtime_diagnostic_task_is_registered():
    runtime_dir = _runtime_test_dir()
    try:
        agent = DummyAgent()
        runtime = RuntimeService(
            agent, MessageBus(),
            diagnostic_interval_seconds=5,
            runtime_dir=runtime_dir,
        )
        task_names = {task.name for task in runtime._tasks}
        assert "diagnostic" in task_names
        assert runtime.heartbeat_path.parent.exists()
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_diagnostic_delivers_error_to_notifier(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    delivered = []

    async def notifier(errors):
        delivered.append(errors)

    fake = _FakeRuntimeSnapshots([
        _diag_snapshot(tasks=[{"name": "heartbeat", "error_count": 0, "last_error": None}]),
        _diag_snapshot(
            heartbeat_count=2,
            tasks=[{"name": "heartbeat", "error_count": 1, "last_error": "akshare timeout"}],
        ),
    ])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path, notifier=notifier)
    # 第一次：无 prev，仅初始化
    await monitor.run()
    # 第二次：task_error 增量 → error 级 → 应送达 notifier
    await monitor.run()

    assert delivered, "error 级 issue 应送达 notifier"
    assert any(issue["code"] == "task_error:heartbeat" for issue in delivered[0])


@pytest.mark.anyio
async def test_diagnostic_warn_does_not_trigger_delivery(tmp_path):
    from finabot.runtime import DiagnosticMonitor

    delivered = []

    async def notifier(errors):
        delivered.append(errors)

    snapshot = _diag_snapshot(bus={"inbound_size": 12, "outbound_size": 0})  # warn 级积压
    fake = _FakeRuntimeSnapshots([snapshot])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path, notifier=notifier)
    await monitor.run()

    assert delivered == [], "warn 级 issue 不应触发告警送达"


@pytest.mark.anyio
async def test_diagnostic_posts_webhook_on_error(monkeypatch, tmp_path):
    from finabot.runtime import DiagnosticMonitor

    class FakeWebhookCallable:
        def __init__(self):
            self.captured = []
        def __call__(self, errors):
            self.captured.append(errors)

    fake_hook = FakeWebhookCallable()
    monkeypatch.setattr(DiagnosticMonitor, "_post_webhook_sync", fake_hook)

    fake = _FakeRuntimeSnapshots([
        _diag_snapshot(bus={"inbound_size": 60, "outbound_size": 0}),
    ])
    monitor = DiagnosticMonitor(fake, log_dir=tmp_path, alert_webhook_url="https://example.com/hook")
    await monitor.run()

    assert fake_hook.captured, "error 级 issue 应触发 webhook 调用"
    assert any(issue["code"] == "bus_backlog:inbound" for issue in fake_hook.captured[0])
