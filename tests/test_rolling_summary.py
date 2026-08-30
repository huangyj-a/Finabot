"""Tests for the cross-turn rolling conversation summary."""

import asyncio

from langchain_core.messages import HumanMessage

from finabot.agents import rolling_summary as rs


def _patch_store(monkeypatch):
    """用内存 dict 替换 working-memory 磁盘读写。"""
    store: dict = {}

    def load(key):
        return store.get(key)

    def save(key, data):
        store[key] = data

    monkeypatch.setattr(rs, "load_working_memory", load)
    monkeypatch.setattr(rs, "save_working_memory", save)
    return store


def test_short_conversation_does_not_update(monkeypatch):
    store = _patch_store(monkeypatch)
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "摘要"

    messages = [HumanMessage(content=f"q{i}") for i in range(5)]
    updated = asyncio.run(rs.update_rolling_summary("s1", messages, fake_llm))

    assert updated is False
    assert calls == []
    assert "s1" not in store


def test_updates_when_middle_grows(monkeypatch):
    store = _patch_store(monkeypatch)
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "中期对话摘要"

    messages = [HumanMessage(content=f"q{i}") for i in range(14)]
    updated = asyncio.run(rs.update_rolling_summary("s2", messages, fake_llm))

    assert updated is True
    assert len(calls) == 1
    state = store["s2"]["rolling_summary"]
    assert state["summary"] == "中期对话摘要"
    assert state["last_summarized_at"] == 14 - rs.TAIL_KEEP


def test_merges_with_existing_summary(monkeypatch):
    store = _patch_store(monkeypatch)
    store["s3"] = {"rolling_summary": {"summary": "旧摘要", "last_summarized_at": 8}}
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "合并后的摘要"

    # 22 条消息：中段 messages[8:-6] 共 8 条，达到 WINDOW 阈值，触发合并更新
    messages = [HumanMessage(content=f"q{i}") for i in range(22)]
    updated = asyncio.run(rs.update_rolling_summary("s3", messages, fake_llm))

    assert updated is True
    assert "旧摘要" in calls[0]
    state = store["s3"]["rolling_summary"]
    assert state["summary"] == "合并后的摘要"
    assert state["last_summarized_at"] == 22 - rs.TAIL_KEEP


def test_llm_failure_returns_false_and_keeps_state(monkeypatch):
    store = _patch_store(monkeypatch)
    store["s4"] = {"rolling_summary": {"summary": "已有摘要", "last_summarized_at": 8}}

    async def boom(prompt):
        raise RuntimeError("network down")

    messages = [HumanMessage(content=f"q{i}") for i in range(22)]
    updated = asyncio.run(rs.update_rolling_summary("s4", messages, boom))

    assert updated is False
    # 失败不破坏已有摘要与进度
    assert store["s4"]["rolling_summary"]["summary"] == "已有摘要"
    assert store["s4"]["rolling_summary"]["last_summarized_at"] == 8


def test_get_rolling_summary(monkeypatch):
    store = _patch_store(monkeypatch)

    assert rs.get_rolling_summary("missing") == ""

    store["s5"] = {"rolling_summary": {"summary": "历史摘要", "last_summarized_at": 8}}
    assert rs.get_rolling_summary("s5") == "历史摘要"


def test_disabled_by_env_skips_update_and_injection(monkeypatch):
    store = _patch_store(monkeypatch)
    monkeypatch.setenv("FINABOT_ROLLING_SUMMARY", "off")
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "摘要"

    messages = [HumanMessage(content=f"q{i}") for i in range(20)]
    updated = asyncio.run(rs.update_rolling_summary("s6", messages, fake_llm))

    assert updated is False
    assert calls == []
    # 注入侧：关闭时不返回摘要
    assert rs.get_rolling_summary("s6") == ""


def test_custom_config_thresholds_update_sooner(monkeypatch):
    store = _patch_store(monkeypatch)
    cfg = rs.RollingSummaryConfig(min_messages=4, window=2, tail_keep=2)
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "摘要"

    # 6 条消息：中段 messages[0:-2] 共 4 条 >= window 2，应触发更新
    messages = [HumanMessage(content=f"q{i}") for i in range(6)]
    updated = asyncio.run(rs.update_rolling_summary("s7", messages, fake_llm, config=cfg))

    assert updated is True
    assert len(calls) == 1
    state = store["s7"]["rolling_summary"]
    assert state["summary"] == "摘要"
    assert state["last_summarized_at"] == 6 - 2


def test_env_thresholds_override_defaults(monkeypatch):
    store = _patch_store(monkeypatch)
    monkeypatch.setenv("FINABOT_ROLLING_SUMMARY_MIN_MESSAGES", "4")
    monkeypatch.setenv("FINABOT_ROLLING_SUMMARY_WINDOW", "2")
    monkeypatch.setenv("FINABOT_ROLLING_SUMMARY_TAIL_KEEP", "2")
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "摘要"

    messages = [HumanMessage(content=f"q{i}") for i in range(6)]
    updated = asyncio.run(rs.update_rolling_summary("s8", messages, fake_llm))

    assert updated is True
    assert len(calls) == 1
    assert store["s8"]["rolling_summary"]["last_summarized_at"] == 4
