import json
import sys
import types
from datetime import datetime

import pandas as pd


def test_history_metrics_exposes_latest_trade_date():
    from finabot.tools.akshare_tools import _internal_history_metrics

    today = datetime.now().strftime("%Y-%m-%d")
    metrics = _internal_history_metrics(
        pd.DataFrame(
            [
                {"日期": "2026-01-01", "收盘": 10.0},
                {"日期": today, "收盘": 12.0},
            ]
        )
    )

    assert metrics["latest_trade_date"] == today
    assert "data_lag_days" in metrics
    assert "is_stale" in metrics


def test_unified_news_tool_formats_direct_news(monkeypatch):
    from finabot.tools.news_tools import get_stock_news_unified

    fake_akshare = types.ModuleType("akshare")
    fake_akshare.stock_news_em = lambda symbol: pd.DataFrame(
        [{"新闻标题": "新易盛订单增长", "发布时间": "2026-05-30", "新闻内容": "公司相关订单增长。"}]
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)

    payload = json.loads(get_stock_news_unified.invoke({"stock_code": "300502", "max_news": 3}))

    assert payload["stock_type"] == "A股"
    assert payload["has_direct_news"] is True
    assert payload["news_scope"] == "stock_direct"
    assert "新易盛订单增长" in payload["news"]
    assert "fetch_time" in payload


def test_unified_news_marks_market_fallback_as_non_direct(monkeypatch):
    from finabot.tools.news_tools import get_stock_news_unified

    fake_akshare = types.ModuleType("akshare")
    fake_akshare.stock_news_em = lambda symbol: pd.DataFrame([])
    fake_akshare.stock_info_global_cls = lambda symbol: pd.DataFrame(
        [{"标题": "市场整体风险偏好回升", "时间": "2026-05-30"}]
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)

    payload = json.loads(
        get_stock_news_unified.invoke({"stock_code": "300502", "max_news": 3})
    )

    assert payload["news_scope"] == "market_general"
    assert payload["has_direct_news"] is False
    assert "市场整体风险偏好回升" in payload["news"]


def test_akshare_cache_uses_extracted_stock_query(monkeypatch):
    import finabot.agents.akshare_cache as cache_module
    import finabot.tools.akshare_tools as akshare_tools

    calls = {"lookup": 0}

    class LookupTool:
        def invoke(self, kwargs):
            calls["lookup"] += 1
            assert kwargs["keyword"] == "新易盛"
            return json.dumps({"sample": [{"代码": "300502", "名称": "新易盛"}]}, ensure_ascii=False)

    class StaticTool:
        def invoke(self, kwargs):
            return "{}"

    monkeypatch.setattr(akshare_tools, "stock_a_lookup", LookupTool())
    monkeypatch.setattr(akshare_tools, "stock_a_spot", StaticTool())
    monkeypatch.setattr(akshare_tools, "stock_a_individual_info", StaticTool())
    monkeypatch.setattr(akshare_tools, "stock_a_snapshot", StaticTool())
    monkeypatch.setattr(akshare_tools, "stock_a_conclusion", StaticTool())

    cache = {}
    first = cache_module.get_cached_akshare_data(cache, "帮我分析未来三个月是否适合持有新易盛")
    second = cache_module.get_cached_akshare_data(cache, "300502")

    assert first is second
    assert calls["lookup"] == 1
    assert first["resolved_symbol"] == "300502"
