"""Tests for the evidence registry helpers."""

from finabot.agents.evidence import (
    evidence_summary,
    register_subagent_evidence,
    register_tool_evidence,
)


def test_register_tool_evidence_extracts_metadata():
    registry = {}
    payload = '{"tool": "stock_a_history", "fetch_time": "2026-05-29T15:00:00", "data_as_of": "2026-05-29", "resolved_symbol": "600519", "rows": 22}'
    source_id = register_tool_evidence(registry, "stock_a_history", payload)
    assert source_id in registry
    meta = registry[source_id]
    assert meta["tool"] == "stock_a_history"
    assert meta["retrieved_at"] == "2026-05-29T15:00:00"
    assert meta["data_as_of"] == "2026-05-29"
    assert meta["resolved_symbol"] == "600519"


def test_register_tool_evidence_records_error():
    registry = {}
    payload = '{"tool": "stock_a_valuation", "error": "no data returned"}'
    source_id = register_tool_evidence(registry, "stock_a_valuation", payload)
    assert registry[source_id]["error"] == "no data returned"


def test_register_tool_evidence_non_json_text():
    registry = {}
    source_id = register_tool_evidence(registry, "calculator", "42")
    assert registry[source_id]["tool"] == "calculator"
    assert registry[source_id]["scope"] == "unknown"


def test_register_news_scope():
    registry = {}
    payload = '{"tool": "get_stock_news_unified", "news_scope": "market_general", "has_direct_news": false, "fetch_time": "2026-05-29"}'
    register_tool_evidence(registry, "get_stock_news_unified", payload)
    assert registry["get_stock_news_unified@0"]["scope"] == "market_general"
    assert registry["get_stock_news_unified@0"]["has_direct_news"] is False


def test_register_subagent_evidence():
    registry = {}
    source_id = register_subagent_evidence(registry, "news_analyst", "报告正文", "2026-05-29")
    assert registry[source_id]["source"] == "subagent:news_analyst"
    assert registry[source_id]["preview"] == "报告正文"


def test_evidence_summary_empty():
    assert "无证据记录" in evidence_summary({})