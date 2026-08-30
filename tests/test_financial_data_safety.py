import json
import math
import os

import pandas as pd


def test_invalid_history_date_returns_explicit_error(monkeypatch):
    import finabot.tools.akshare_tools as tools

    monkeypatch.setattr(tools, "_internal_resolve_a_stock_symbol", lambda symbol: "000001")
    payload = json.loads(
        tools.stock_a_history.invoke(
            {
                "symbol": "000001",
                "start_date": "2026-02-30",
                "end_date": "2026-03-01",
            }
        )
    )

    assert payload["tool"] == "stock_a_history"
    assert "invalid calendar date" in payload["error"]


def test_keyword_filter_treats_user_input_as_literal_text():
    from finabot.tools.akshare_tools import _internal_keyword_mask

    frame = pd.DataFrame(
        [
            {"代码": "000001", "名称": "平安(银行"},
            {"代码": "000002", "名称": "万科A"},
        ]
    )

    filtered = _internal_keyword_mask(frame, "(")

    assert filtered["代码"].tolist() == ["000001"]


def test_keyword_filter_returns_empty_frame_instead_of_unrelated_rows():
    from finabot.tools.akshare_tools import _internal_keyword_mask

    frame = pd.DataFrame(
        [
            {"代码": "000001", "名称": "平安银行"},
            {"代码": "000002", "名称": "万科A"},
        ]
    )

    filtered = _internal_keyword_mask(frame, "完全不存在")

    assert filtered.empty


def test_dataframe_payload_replaces_non_finite_numbers_with_null():
    from finabot.tools.akshare_tools import _internal_dataframe_payload

    payload_text = _internal_dataframe_payload(
        "finite_test",
        pd.DataFrame([{"正常": 1.2, "缺失": math.nan, "无穷": math.inf}]),
    )
    payload = json.loads(payload_text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    assert payload["sample"] == [{"正常": 1.2, "缺失": None, "无穷": None}]
    assert "NaN" not in payload_text
    assert "Infinity" not in payload_text


def test_single_row_formatter_does_not_fallback_to_another_stock():
    from finabot.tools.akshare_tools import _internal_format_single_row

    payload = _internal_format_single_row(
        pd.DataFrame(
            [
                {"代码": "000001", "名称": "平安银行"},
                {"代码": "000002", "名称": "万科A"},
            ]
        ),
        "stock_a_spot",
        "600519",
    )

    assert payload["rows"] == 0
    assert payload["sample"] == []


def test_history_metrics_ignore_non_finite_close_values():
    from finabot.tools.akshare_tools import _internal_history_metrics

    metrics = _internal_history_metrics(
        pd.DataFrame(
            [
                {"日期": "2026-01-01", "收盘": 10.0},
                {"日期": "2026-01-02", "收盘": math.nan},
                {"日期": "2026-01-03", "收盘": math.inf},
                {"日期": "2026-01-04", "收盘": 12.0},
            ]
        )
    )

    assert metrics["rows"] == 2
    assert metrics["latest_close"] == 12.0
    assert metrics["high"] == 12.0
    assert metrics["low"] == 10.0
    assert metrics["latest_trade_date"] == "2026-01-04"


def test_akshare_network_wrapper_does_not_mutate_proxy_environment(monkeypatch):
    import finabot.tools.akshare_tools as tools

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    before = dict(os.environ)
    with tools._internal_without_proxy_env():
        assert os.environ["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert dict(os.environ) == before


def test_calculator_evaluates_safe_arithmetic():
    from finabot.tools.base import calculator

    assert calculator.invoke({"expression": "1 + 2 * 3"}) == "7"
    assert calculator.invoke({"expression": "(10 - 4) / 2"}) == "3.0"


def test_calculator_rejects_resource_exhaustion_expression():
    from finabot.tools.base import calculator

    assert calculator.invoke({"expression": "2 ** 99999999"}) == "计算错误"
    assert calculator.invoke({"expression": "2 ** 1000000"}) == "计算错误"
    # 边界：正常幂仍可用
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"
