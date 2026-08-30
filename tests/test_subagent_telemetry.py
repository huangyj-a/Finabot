"""Tests for per-sub-agent telemetry."""

import asyncio

from finabot.agents.telemetry import SUBAGENT_METRICS, snapshot_subagent_metrics
from finabot.agents.nodes import _call_with_timeout


def test_call_with_timeout_records_success():
    SUBAGENT_METRICS.reset()

    async def quick():
        return "ok"

    result = asyncio.run(_call_with_timeout(quick(), "news_analyst", timeout=5.0))
    assert result == "ok"

    snap = snapshot_subagent_metrics()
    assert snap["subagents"]["news_analyst"]["calls"] == 1
    assert snap["subagents"]["news_analyst"]["failures"] == 0
    assert snap["subagents"]["news_analyst"]["average_latency_ms"] >= 0


def test_call_with_timeout_records_timeout():
    SUBAGENT_METRICS.reset()

    async def slow():
        await asyncio.sleep(1.0)
        return "late"

    result = asyncio.run(_call_with_timeout(slow(), "market_analyst", timeout=0.01))
    assert result.startswith("[subagent_timeout")

    snap = snapshot_subagent_metrics()
    assert snap["subagents"]["market_analyst"]["calls"] == 1
    assert snap["subagents"]["market_analyst"]["failures"] == 1


def test_snapshot_empty_registry():
    SUBAGENT_METRICS.reset()
    snap = snapshot_subagent_metrics()
    assert snap["subagents"] == {}
    assert snap["recent"] == []