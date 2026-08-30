"""Tests for per-run trace recording."""

import json

from finabot.agents.trace import _safe_name, write_run_trace


def test_safe_name_sanitizes_session_key():
    assert _safe_name("cli:direct") == "cli_direct"
    assert _safe_name("wx:123/45") == "wx_123_45"
    assert _safe_name("") == "default"


def test_write_run_trace_creates_json(tmp_path, monkeypatch):
    monkeypatch.setenv("FINABOT_RUNTIME_DIR", str(tmp_path / "runtime"))
    state = {
        "messages": [{"type": "ai", "content": "结论"}],
        "market_report": "市场",
        "news_report": "新闻",
        "bull_report": "看涨",
        "bear_report": "看跌",
        "fundamentals_report": "基本面",
        "evidence_registry": {"s0": {"tool": "x"}},
        "risk_flags": ["估值高"],
        "claims": [{"text": "主张"}],
        "run_meta": {"llm_calls": 3},
    }
    path = write_run_trace("cli:direct", state)
    assert path is not None
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session_key"] == "cli:direct"
    assert payload["reports"]["news"] == "新闻"
    assert payload["risk_flags"] == ["估值高"]


def test_write_run_trace_best_effort_on_empty():
    path = write_run_trace("s", None)
    # 空状态也应能落盘（默认空字段），或至少不抛异常
    assert path is None or path.exists()