import json

import pandas as pd

from finabot.agents.nodes import format_tools
from finabot.tools.base import get_tools
import finabot.tools.akshare_tools as aktools


def _internal_sample_frame():
    return pd.DataFrame(
        [
            {"代码": "000001", "名称": "平安银行", "最新价": 10.1},
            {"代码": "000002", "名称": "万科A", "最新价": 8.2},
        ]
    )


def test_stock_a_history_returns_structured_json(monkeypatch):
    monkeypatch.setattr(aktools.ak, "stock_zh_a_hist", lambda **kwargs: _internal_sample_frame())

    payload = json.loads(
        aktools.stock_a_history.invoke(
            {"symbol": "000001", "start_date": "20260501", "end_date": "20260529"}
        )
    )

    assert payload["tool"] == "stock_a_history"
    assert payload["rows"] == 2
    assert payload["sample"][0]["代码"] == "000001"
    assert payload["sample"][0]["名称"] == "平安银行"


def test_stock_a_history_resolves_name_to_code(monkeypatch):
    captured = {}

    def fake_hist(**kwargs):
        captured.update(kwargs)
        return _internal_sample_frame()

    monkeypatch.setattr(aktools.ak, "stock_zh_a_hist", fake_hist)
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "600519", "证券简称": "贵州茅台"},
                {"证券代码": "000001", "证券简称": "平安银行"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_history.invoke({"symbol": "贵州茅台"}))

    assert captured["symbol"] == "600519"
    assert payload["tool"] == "stock_a_history"


def test_stock_a_snapshot_resolves_name_and_collects_sections(monkeypatch):
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "600519", "证券简称": "贵州茅台"},
                {"证券代码": "000001", "证券简称": "平安银行"},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_a_spot_em",
        lambda: pd.DataFrame(
            [
                {"代码": "600519", "名称": "贵州茅台", "最新价": 1700},
                {"代码": "000001", "名称": "平安银行", "最新价": 10},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_a_hist",
        lambda **kwargs: pd.DataFrame(
            [
                {"日期": "2026-05-29", "收盘": 1700},
                {"日期": "2026-05-28", "收盘": 1690},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_individual_info_em",
        lambda symbol=None, timeout=None: pd.DataFrame(
            [
                {"item": "总市值", "value": "2000亿"},
                {"item": "市盈率", "value": "40"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_snapshot.invoke({"symbol_or_name": "贵州茅台", "history_days": 30}))

    assert payload["tool"] == "stock_a_snapshot"
    assert payload["resolved_symbol"] == "600519"
    assert payload["resolved_name"] == "贵州茅台"
    assert payload["spot"]["tool"] == "stock_a_spot"
    assert payload["history"]["tool"] == "stock_a_history"
    assert payload["profile"]["tool"] == "stock_a_individual_info"


def test_stock_a_hold_analysis_returns_rule_based_view(monkeypatch):
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "688256", "证券简称": "寒武纪"},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_a_spot_em",
        lambda: pd.DataFrame(
            [
                {"代码": "688256", "名称": "寒武纪", "最新价": 500},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_a_hist",
        lambda **kwargs: pd.DataFrame(
            [
                {"日期": "2026-03-01", "收盘": 200},
                {"日期": "2026-04-01", "收盘": 250},
                {"日期": "2026-05-01", "收盘": 300},
                {"日期": "2026-05-29", "收盘": 400},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_individual_info_em",
        lambda symbol=None, timeout=None: pd.DataFrame(
            [
                {"item": "总市值", "value": "1000亿"},
                {"item": "行业", "value": "半导体"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_hold_analysis.invoke({"symbol_or_name": "寒武纪", "history_days": 90}))

    assert payload["tool"] == "stock_a_hold_analysis"
    assert payload["resolved_symbol"] == "688256"
    assert payload["hold_view"] in {"偏向持有", "谨慎持有", "中性观察"}
    assert payload["metrics"]["latest_close"] == 400.0
    assert payload["evidence"]
    assert payload["snapshot"]["tool"] == "stock_a_spot"
    assert payload["history"]["tool"] == "stock_a_history"
    assert payload["profile"]["tool"] == "stock_a_individual_info"


def test_stock_a_conclusion_is_front_loaded(monkeypatch):
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "688256", "证券简称": "寒武纪"},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_a_spot_em",
        lambda: pd.DataFrame(
            [
                {"代码": "688256", "名称": "寒武纪", "最新价": 500},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_a_hist",
        lambda **kwargs: pd.DataFrame(
            [
                {"日期": "2026-03-01", "收盘": 200},
                {"日期": "2026-04-01", "收盘": 250},
                {"日期": "2026-05-01", "收盘": 300},
                {"日期": "2026-05-29", "收盘": 400},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_individual_info_em",
        lambda symbol=None, timeout=None: pd.DataFrame(
            [
                {"item": "总市值", "value": "1000亿"},
                {"item": "行业", "value": "半导体"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_conclusion.invoke({"symbol_or_name": "寒武纪", "history_days": 90}))

    assert payload["tool"] == "stock_a_conclusion"
    assert payload["conclusion"] in {"偏向持有", "谨慎持有", "中性观察"}
    assert payload["confidence"] in {"low", "medium"}
    assert len(payload["evidence"]) >= 2


def test_stock_a_individual_info_resolves_name(monkeypatch):
    captured = {}

    def fake_info(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {"item": "股票代码", "value": "600519"},
                {"item": "股票简称", "value": "贵州茅台"},
            ]
        )

    monkeypatch.setattr(aktools.ak, "stock_individual_info_em", fake_info)
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "600519", "证券简称": "贵州茅台"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_individual_info.invoke({"symbol_or_name": "贵州茅台"}))

    assert captured["symbol"] == "600519"
    assert payload["tool"] == "stock_a_individual_info"
    assert any(row["item"] == "股票简称" for row in payload["sample"])


def test_stock_a_lookup_returns_matches(monkeypatch):
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "600519", "证券简称": "贵州茅台"},
                {"证券代码": "000858", "证券简称": "五粮液"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_lookup.invoke({"keyword": "茅台"}))

    assert payload["tool"] == "stock_a_lookup"
    assert any(row["证券简称"] == "贵州茅台" for row in payload["sample"])


def test_stock_a_lookup_returns_empty_sample_when_keyword_does_not_match(monkeypatch):
    monkeypatch.setattr(
        aktools.ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            [
                {"证券代码": "600519", "证券简称": "贵州茅台"},
                {"证券代码": "000858", "证券简称": "五粮液"},
            ]
        ),
    )

    payload = json.loads(aktools.stock_a_lookup.invoke({"keyword": "完全不存在的股票"}))

    assert payload["tool"] == "stock_a_lookup"
    assert payload["rows"] == 0
    assert payload["sample"] == []


def test_market_summary_routes_to_selected_exchange(monkeypatch):
    calls = {"sse": 0, "szse": 0}

    def fake_sse():
        calls["sse"] += 1
        return _internal_sample_frame()

    def fake_szse(date=None):
        calls["szse"] += 1
        return _internal_sample_frame()

    monkeypatch.setattr(aktools.ak, "stock_sse_summary", fake_sse)
    monkeypatch.setattr(aktools.ak, "stock_szse_summary", fake_szse)

    sse_payload = json.loads(aktools.market_summary.invoke({"exchange": "sse"}))
    szse_payload = json.loads(aktools.market_summary.invoke({"exchange": "szse", "date": "20260529"}))

    assert calls == {"sse": 1, "szse": 1}
    assert sse_payload["tool"] == "market_summary_sse"
    assert szse_payload["tool"] == "market_summary_szse"


def test_fund_and_index_tools_return_json(monkeypatch):
    monkeypatch.setattr(aktools.ak, "stock_zh_index_spot_em", lambda symbol="沪深重要指数": _internal_sample_frame())
    monkeypatch.setattr(
        aktools.ak,
        "index_zh_a_hist",
        lambda **kwargs: pd.DataFrame(
            [
                {"日期": "2026-05-29", "收盘": 3000},
                {"日期": "2026-05-28", "收盘": 2990},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "index_zh_a_hist_min_em",
        lambda **kwargs: pd.DataFrame(
            [
                {"时间": "2026-05-29 09:30:00", "收盘": 3000},
                {"时间": "2026-05-29 09:31:00", "收盘": 3001},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_zh_index_spot_sina",
        lambda: pd.DataFrame(
            [
                {"名称": "上证指数", "最新价": 3000},
                {"名称": "深证成指", "最新价": 9000},
                {"名称": "其他指数", "最新价": 1234},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_hk_index_spot_em",
        lambda: pd.DataFrame(
            [
                {"名称": "恒生指数", "最新价": 18000},
                {"名称": "恒生科技指数", "最新价": 4200},
                {"名称": "其他港股指数", "最新价": 1000},
            ]
        ),
    )
    monkeypatch.setattr(
        aktools.ak,
        "stock_hk_index_daily_em",
        lambda symbol=None: pd.DataFrame(
            [
                {"date": "2026-05-29", "latest": 18000},
                {"date": "2026-05-28", "latest": 17950},
            ]
        ),
    )
    monkeypatch.setattr(aktools.ak, "fund_etf_spot_em", lambda: _internal_sample_frame())
    monkeypatch.setattr(aktools.ak, "fund_open_fund_daily_em", lambda: _internal_sample_frame())
    monkeypatch.setattr(aktools.ak, "fund_etf_fund_daily_em", lambda: _internal_sample_frame())
    monkeypatch.setattr(aktools.ak, "fund_money_fund_daily_em", lambda: _internal_sample_frame())

    assert json.loads(aktools.index_spot.invoke({"symbol": "沪深重要指数", "keyword": "平安"}))["tool"] == "index_spot"
    assert json.loads(aktools.index_history.invoke({"symbol": "000016"}))["tool"] == "index_history"
    assert json.loads(aktools.index_minute.invoke({"symbol": "000001"}))["tool"] == "index_minute"
    classic_payload = json.loads(aktools.index_classic_spot.invoke({"keyword": "上证"}))
    assert classic_payload["tool"] == "index_classic_spot"
    assert any(row["名称"] == "上证指数" for row in classic_payload["sample"])
    hk_payload = json.loads(aktools.hk_index_spot.invoke({"keyword": "恒生"}))
    assert hk_payload["tool"] == "hk_index_spot"
    assert any(row["名称"] == "恒生指数" for row in hk_payload["sample"])
    assert json.loads(aktools.hk_index_history.invoke({"symbol": "HSTECF2L"}))["tool"] == "hk_index_history"
    assert json.loads(aktools.fund_etf_spot.invoke({"keyword": "平安"}))["tool"] == "fund_etf_spot"
    assert json.loads(aktools.fund_open_daily.invoke({"keyword": "平安"}))["tool"] == "fund_open_daily"
    assert json.loads(aktools.fund_etf_daily.invoke({"keyword": "平安"}))["tool"] == "fund_etf_daily"
    assert json.loads(aktools.fund_money_daily.invoke({"keyword": "平安"}))["tool"] == "fund_money_daily"
    assert json.loads(aktools.fund_index_spot.invoke({"keyword": "上证"}))["tool"] == "fund_index_spot"


def test_akshare_tools_are_registered_and_schema_is_dynamic():
    tool_names = {tool.name for tool in get_tools()}

    assert {"stock_a_history", "stock_a_snapshot", "stock_a_hold_analysis", "stock_a_conclusion", "stock_a_individual_info", "stock_a_lookup", "stock_a_spot", "market_summary", "index_spot", "index_history", "index_minute", "index_classic_spot", "hk_index_spot", "hk_index_history", "fund_etf_spot", "fund_open_daily", "fund_etf_daily", "fund_money_daily", "fund_index_spot"}.issubset(tool_names)

    history_schema = next(item for item in format_tools() if item["function"]["name"] == "stock_a_history")
    assert set(history_schema["function"]["parameters"]["required"]) == {"symbol"}
    assert "start_date" in history_schema["function"]["parameters"]["properties"]
    assert "end_date" in history_schema["function"]["parameters"]["properties"]


def test_tool_returns_error_json_when_akshare_fails(monkeypatch):
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(aktools.ak, "stock_zh_a_spot_em", boom)

    payload = json.loads(aktools.stock_a_spot.invoke({"keyword": "平安"}))

    assert payload["tool"] == "stock_a_spot"
    assert payload["error"] == "boom"