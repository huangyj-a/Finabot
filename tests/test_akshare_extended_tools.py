import json

import pandas as pd


def test_extended_akshare_tools_registered():
    from finabot.tools.akshare_tools import get_akshare_tools

    names = {tool.name for tool in get_akshare_tools()}

    assert "stock_a_valuation" in names
    assert "stock_a_financial_indicators" in names
    assert "stock_a_fund_flow" in names
    assert "stock_a_research_report" in names
    assert "stock_a_notice" in names


def test_stock_a_valuation_payload_has_fetch_metadata(monkeypatch):
    import finabot.tools.akshare_tools as tools

    monkeypatch.setattr(tools, "_internal_resolve_a_stock_symbol", lambda symbol: "300502")
    monkeypatch.setattr(
        tools.ak,
        "stock_zh_valuation_baidu",
        lambda symbol: pd.DataFrame([{"日期": "2026-05-29", "市盈率TTM": 65.51}]),
        raising=False,
    )

    payload = json.loads(tools.stock_a_valuation.invoke({"symbol_or_name": "新易盛"}))

    assert payload["resolved_symbol"] == "300502"
    assert payload["data_as_of"] == "2026-05-29"
    assert "fetch_time" in payload
    assert payload["sample"][0]["市盈率TTM"] == 65.51


def test_akshare_cache_contains_extended_fields(monkeypatch):
    import finabot.agents.akshare_cache as cache_module
    import finabot.tools.akshare_tools as tools

    class StaticTool:
        def __init__(self, value):
            self.value = value

        def invoke(self, kwargs):
            return self.value

    monkeypatch.setattr(tools, "stock_a_lookup", StaticTool(json.dumps({"candidates": [{"代码": "300502", "名称": "新易盛"}]}, ensure_ascii=False)))
    monkeypatch.setattr(tools, "stock_a_spot", StaticTool("spot"))
    monkeypatch.setattr(tools, "stock_a_individual_info", StaticTool("info"))
    monkeypatch.setattr(tools, "stock_a_snapshot", StaticTool("snapshot"))
    monkeypatch.setattr(tools, "stock_a_conclusion", StaticTool("conclusion"))
    monkeypatch.setattr(tools, "stock_a_valuation", StaticTool("valuation"))
    monkeypatch.setattr(tools, "stock_a_financial_indicators", StaticTool("financial"))
    monkeypatch.setattr(tools, "stock_a_fund_flow", StaticTool("fund_flow"))
    monkeypatch.setattr(tools, "stock_a_research_report", StaticTool("research"))
    monkeypatch.setattr(tools, "stock_a_notice", StaticTool("notice"))

    payload = cache_module.get_cached_akshare_data({}, "新易盛")

    assert payload["stock_valuation"] == "valuation"
    assert payload["stock_financial_indicators"] == "financial"
    assert payload["stock_fund_flow"] == "fund_flow"
    assert payload["stock_research_report"] == "research"
    assert payload["stock_notice"] == "notice"
