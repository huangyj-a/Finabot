"""Background scheduling and heartbeat support for Finabot."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
        runtime_dir: Path | None = None,
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
        """Clean expired in-memory sessions without touching persisted memory."""

        session_manager = getattr(self.agent, "session_manager", None)
        if session_manager is None:
            return
        expired_keys = session_manager.cleanup_expired()
        sessions = getattr(self.agent, "sessions", None)
        if isinstance(sessions, dict):
            for key in expired_keys:
                sessions.pop(key, None)

    async def write_heartbeat(self) -> None:
        self.heartbeat_count += 1
        payload = self.snapshot()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        sessions = getattr(self.agent, "sessions", {})
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
                "active_sessions": len(sessions) if isinstance(sessions, dict) else 0,
                "active_message_tasks": len(active_tasks) if isinstance(active_tasks, set) else 0,
            },
            "tasks": [task.snapshot() for task in self._tasks],
        }

