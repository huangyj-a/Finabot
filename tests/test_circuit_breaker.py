"""Tests for the LLM circuit breaker."""

import asyncio

from finabot.agents.telemetry import LLMCircuitBreaker


def test_circuit_breaker_starts_closed():
    cb = LLMCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
    assert cb.is_open() is False


def test_circuit_breaker_opens_after_threshold():
    cb = LLMCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open() is False  # 未达阈值
    cb.record_failure()
    assert cb.is_open() is True  # 达到阈值，熔断打开


def test_circuit_breaker_success_closes():
    cb = LLMCircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open() is True
    cb.record_success()
    assert cb.is_open() is False


def test_circuit_breaker_cooldown_half_open():
    cb = LLMCircuitBreaker(failure_threshold=2, cooldown_seconds=0.01)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open() is True
    # 冷却结束后半开（重置计数），is_open 返回 False
    import time
    time.sleep(0.02)
    assert cb.is_open() is False


def test_litellm_glm_call_raises_when_open(monkeypatch):
    import finabot.agents.llm as llm_module
    from finabot.agents.telemetry import LLM_CIRCUIT_BREAKER

    LLM_CIRCUIT_BREAKER.record_failure()
    LLM_CIRCUIT_BREAKER.record_failure()
    LLM_CIRCUIT_BREAKER.record_failure()
    LLM_CIRCUIT_BREAKER.record_failure()
    LLM_CIRCUIT_BREAKER.record_failure()

    async def _run():
        with __import__("pytest").raises(RuntimeError, match="熔断"):
            await llm_module.litellm_glm_call([])

    asyncio.run(_run())
    LLM_CIRCUIT_BREAKER.record_success()  # 清理状态