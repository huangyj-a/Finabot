"""Tests for run-level observability: shadow mode + LLM call counting."""

import asyncio
from types import SimpleNamespace

from finabot.eval.frozen_data import shadow_mode_enabled


def test_shadow_mode_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FINABOT_EVAL_SHADOW", raising=False)
    assert shadow_mode_enabled() is False


def test_shadow_mode_enabled(monkeypatch):
    monkeypatch.setenv("FINABOT_EVAL_SHADOW", "1")
    assert shadow_mode_enabled() is True


def test_call_llm_node_increments_llm_calls(monkeypatch):
    import finabot.agents.nodes as nodes_module

    async def fake_glm(messages=None, tools=None, memories=None, stream_label=None, system_prompt=None, **kwargs):
        return SimpleNamespace(content="最终答案", tool_calls=[])

    monkeypatch.setattr(nodes_module, "litellm_glm_call", fake_glm)

    result = asyncio.run(
        nodes_module.call_llm_node({"messages": [], "memories": [], "run_meta": {"llm_calls": 2}})
    )
    assert result["run_meta"]["llm_calls"] == 3
    assert result["messages"][0].content == "最终答案"


def test_call_llm_node_defaults_llm_calls_from_zero(monkeypatch):
    import finabot.agents.nodes as nodes_module

    async def fake_glm(messages=None, tools=None, memories=None, stream_label=None, system_prompt=None, **kwargs):
        return SimpleNamespace(content="答案", tool_calls=[])

    monkeypatch.setattr(nodes_module, "litellm_glm_call", fake_glm)

    result = asyncio.run(
        nodes_module.call_llm_node({"messages": [], "memories": []})
    )
    assert result["run_meta"]["llm_calls"] == 1