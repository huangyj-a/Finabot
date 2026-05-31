"""Tests for background runtime scheduling and heartbeat."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from finabot.agents.session import SessionManager
from finabot.bus.queue import MessageBus
from finabot.runtime import PeriodicTask, RuntimeService


def _runtime_test_dir() -> Path:
    path = Path("memory") / f"runtime_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class DummyAgent:
    def __init__(self):
        self.sessions = {}
        self._tasks = set()
        self.session_manager = SessionManager(ttl_minutes=1)


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
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_runtime_maintenance_removes_expired_sessions():
    runtime_dir = _runtime_test_dir()
    try:
        agent = DummyAgent()
        key = "cli:expired"
        agent.sessions[key] = {"messages": []}
        agent.session_manager.sessions[key] = []
        agent.session_manager.last_access[key] = datetime.now() - timedelta(minutes=5)
        runtime = RuntimeService(agent, MessageBus(), runtime_dir=runtime_dir)

        await runtime.run_maintenance()

        assert key not in agent.sessions
        assert key not in agent.session_manager.sessions
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
