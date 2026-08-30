"""AKShare-backed market data tools."""

from __future__ import annotations

import contextlib
import json
import math
import re
from datetime import datetime, timedelta
from numbers import Real

import akshare as ak
from langchain_core.tools import tool
from pandas.api.types import is_string_dtype

from finabot.utils.clock import now as clock_now


_CLASSIC_INDEX_NAMES = [
    "上证指数",
    "深证成指",
    "创业板指",
    "科创50",
    "沪深300",
    "上证50",
    "中证500",
    "中证1000",
    "北证50",
    "深证100",
    "上证180",
    "中证2000",
    "中证全指",
    "创业板50",
    "上证科创板50",
    "中证A50",
    "中证A500",
    "中证800",
    "中小100",
    "富时A50指数",
    "国债指数",
    "沪企债指数",
    "深企债指数",
    "上证国债指数",
    "中证国债指数",
    "中证红利",
    "深证红利",
    "沪深300价值",
    "中证医疗",
    "中证消费",
]

_CLASSIC_HK_INDEX_NAMES = [
    "恒生指数",
    "恒生科技指数",
    "恒生中国企业指数",
    "恒生香港中资企业指数",
    "恒生互联网科技业指数",
    "恒生医疗保健指数",
    "恒生金融业指数",
    "恒生地产建筑业指数",
    "恒生消费品制造及服务业指数",
    "恒生高股息率指数",
]

@contextlib.contextmanager
def _internal_without_proxy_env():
    """Compatibility wrapper that deliberately avoids global network mutation.

    AKShare and requests inherit the caller's normal proxy configuration. Older
    versions of this helper temporarily rewrote process environment variables,
    requests module functions, and AKShare global proxy state, which was unsafe
    when different sessions fetched market data concurrently.
    """

    yield


def _internal_normalize_date(value: str | None) -> str:
    if value is None or not str(value).strip():
        return clock_now().strftime("%Y%m%d")

    cleaned = re.sub(r"[^0-9]", "", str(value))
    if len(cleaned) != 8:
        raise ValueError(f"invalid date: {value}; expected YYYYMMDD or YYYY-MM-DD")

    try:
        parsed = datetime.strptime(cleaned, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid calendar date: {value}") from exc
    return parsed.strftime("%Y%m%d")


def _internal_keyword_mask(df, keyword: str | None):
    if not keyword or df is None or getattr(df, "empty", True):
        return df

    keyword_text = str(keyword).strip()
    if not keyword_text:
        return df

    text_columns = []
    for column in getattr(df, "columns", []):
        try:
            series = df[column]
            if is_string_dtype(getattr(series, "dtype", None)):
                text_columns.append(column)
            elif any(token in str(column).lower() for token in ["name", "code", "symbol", "代码", "名称", "代码"]):
                text_columns.append(column)
        except Exception:
            continue

    if not text_columns:
        return df

    mask = None
    for column in text_columns:
        try:
            column_mask = df[column].astype(str).str.contains(
                keyword_text,
                case=False,
                na=False,
                regex=False,
            )
        except Exception:
            continue
        mask = column_mask if mask is None else (mask | column_mask)

    if mask is None:
        return df

    return df[mask]


def _internal_name_filter(df, target_names: list[str], keyword: str | None = None, top_n: int = 20):
    if df is None or getattr(df, "empty", True):
        return df

    if "名称" not in getattr(df, "columns", []):
        return df

    filtered = df[df["名称"].isin(target_names)]
    if keyword:
        keyword_text = str(keyword).strip()
        if keyword_text:
            filtered = filtered[
                filtered.astype(str).apply(
                    lambda row: row.str.contains(
                keyword_text,
                case=False,
                na=False,
                regex=False,
            ).any(),
                    axis=1,
                )
            ]

    if getattr(filtered, "empty", True):
        return df.head(max(1, min(int(top_n or 20), 50)))

    return filtered.head(max(1, min(int(top_n or 20), 50)))


def _internal_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _internal_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_internal_json_safe(item) for item in value]
    if isinstance(value, Real):
        try:
            if not math.isfinite(float(value)):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _internal_dataframe_payload(tool_name: str, df, keyword: str | None = None, top_n: int = 20) -> str:
    if df is None:
        return json.dumps({"tool": tool_name, "error": "no data returned"}, ensure_ascii=False)

    filtered = _internal_keyword_mask(df, keyword)
    limit = max(1, min(int(top_n or 20), 50))
    sample = filtered.head(limit).to_dict(orient="records") if not getattr(filtered, "empty", True) else []
    sample = _internal_json_safe(sample)

    payload = {
        "tool": tool_name,
        "rows": int(getattr(filtered, "shape", [0])[0]),
        "columns": list(getattr(filtered, "columns", [])),
        "sample": sample,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _internal_safe_fetch(tool_name: str, fetcher, keyword: str | None = None, top_n: int = 20) -> str:
    try:
        with _internal_without_proxy_env():
            df = fetcher()
        return _internal_dataframe_payload(tool_name, df, keyword=keyword, top_n=top_n)
    except Exception as exc:
        return json.dumps({"tool": tool_name, "error": str(exc)}, ensure_ascii=False)


def _internal_format_single_row(df, tool_name: str, symbol: str, top_n: int = 5) -> dict:
    if df is None or getattr(df, "empty", True):
        return {"tool": tool_name, "symbol": symbol, "rows": 0, "columns": [], "sample": []}

    filtered = _internal_keyword_mask(df, symbol)

    return {
        "tool": tool_name,
        "symbol": symbol,
        "rows": int(getattr(filtered, "shape", [0])[0]),
        "columns": list(getattr(filtered, "columns", [])),
        "sample": _internal_json_safe(
            filtered.head(max(1, min(int(top_n or 5), 10))).to_dict(orient="records")
        ),
    }


def _internal_pick_column(columns, candidates: list[str]) -> str | None:
    for candidate in candidates:
        for column in columns:
            if candidate.lower() == str(column).lower():
                return column
    for candidate in candidates:
        for column in columns:
            if candidate in str(column):
                return column
    return None


def _internal_lookup_columns(df) -> tuple[str | None, str | None]:
    columns = list(getattr(df, "columns", []))
    code_column = _internal_pick_column(columns, ["code", "代码", "证券代码"])
    name_column = _internal_pick_column(columns, ["name", "简称", "名称", "证券简称"])

    if code_column is None or name_column is None:
        object_columns = [
            col for col in columns if is_string_dtype(getattr(df[col], "dtype", None))
        ]
        if code_column is None and object_columns:
            code_column = object_columns[0]
        if name_column is None and len(object_columns) > 1:
            name_column = object_columns[1]

    return code_column, name_column


def _internal_history_metrics(history_df):
    if history_df is None or getattr(history_df, "empty", True):
        return {"error": "no history data"}

    columns = list(getattr(history_df, "columns", []))
    close_column = _internal_pick_column(columns, ["收盘", "close", "latest"])
    date_column = _internal_pick_column(columns, ["日期", "date", "时间"])

    if close_column is None:
        return {"error": "missing close column"}

    frame = history_df.copy()
    if date_column is not None:
        try:
            frame = frame.sort_values(by=date_column)
        except Exception:
            pass

    closes = []
    valid_positions = []
    for position, raw_value in enumerate(frame[close_column].tolist()):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            closes.append(value)
            valid_positions.append(position)
    if not closes:
        return {"error": "empty finite close series"}

    frame = frame.iloc[valid_positions]
    latest_close = float(closes[-1])
    metrics: dict[str, float | str | int | None] = {
        "rows": len(closes),
        "latest_close": latest_close,
        "high": max(closes),
        "low": min(closes),
    }
    if date_column is not None:
        latest_trade_date = str(frame.iloc[-1][date_column])
        metrics["latest_trade_date"] = latest_trade_date
        try:
            parsed_trade_date = datetime.strptime(re.sub(r"[^0-9]", "", latest_trade_date)[:8], "%Y%m%d")
            metrics["data_lag_days"] = (clock_now() - parsed_trade_date).days
            metrics["is_stale"] = metrics["data_lag_days"] > 7
        except Exception:
            metrics["data_lag_days"] = None
            metrics["is_stale"] = None

    def _internal_return(period: int) -> float | None:
        if len(closes) <= period:
            return None
        base = float(closes[-period - 1])
        if base == 0:
            return None
        return round((latest_close / base - 1) * 100, 2)

    def _internal_ma(period: int) -> float | None:
        if len(closes) < period:
            return None
        subset = closes[-period:]
        return round(sum(float(item) for item in subset) / period, 2)

    metrics.update(
        {
            "return_20d_pct": _internal_return(20),
            "return_60d_pct": _internal_return(60),
            "ma_20": _internal_ma(20),
            "ma_60": _internal_ma(60),
        }
    )

    if metrics["ma_20"] is not None and metrics["ma_60"] is not None:
        if metrics["latest_close"] >= metrics["ma_20"] >= metrics["ma_60"]:
            view = "trend_positive"
        elif metrics["latest_close"] < metrics["ma_20"] <= metrics["ma_60"]:
            view = "trend_negative"
        else:
            view = "trend_mixed"
    elif metrics["return_20d_pct"] is not None and metrics["return_20d_pct"] > 0:
        view = "trend_positive"
    elif metrics["return_20d_pct"] is not None and metrics["return_20d_pct"] < 0:
        view = "trend_negative"
    else:
        view = "trend_mixed"

    metrics["view"] = view
    return metrics


def _internal_build_hold_conclusion(metrics, info_payload, snapshot_payload):
    if not isinstance(metrics, dict) or metrics.get("error"):
        return {
            "conclusion": "数据不足，暂不建议直接下结论",
            "confidence": "low",
            "reason": "历史行情或关键指标缺失，无法形成可靠判断。",
        }

    view = metrics.get("view", "trend_mixed")
    if view == "trend_positive":
        conclusion = "偏向持有"
        confidence = "medium"
    elif view == "trend_negative":
        conclusion = "谨慎持有"
        confidence = "medium"
    else:
        conclusion = "中性观察"
        confidence = "low"

    evidence = []
    latest_trade_date = metrics.get("latest_trade_date")
    if latest_trade_date is not None:
        evidence.append(f"行情最后交易日: {latest_trade_date}")
    latest_close = metrics.get("latest_close")
    if latest_close is not None:
        evidence.append(f"最新收盘价: {latest_close}")

    for key, label in [
        ("return_20d_pct", "近20日涨跌幅"),
        ("return_60d_pct", "近60日涨跌幅"),
        ("ma_20", "20日均线"),
        ("ma_60", "60日均线"),
    ]:
        value = metrics.get(key)
        if value is not None:
            evidence.append(f"{label}: {value}")

    if isinstance(info_payload, dict):
        info_sample = info_payload.get("sample", [])
        if info_sample:
            for row in info_sample[:3]:
                item = row.get("item")
                value = row.get("value")
                if item and value is not None:
                    evidence.append(f"{item}: {value}")

    if isinstance(snapshot_payload, dict):
        snapshot_sample = snapshot_payload.get("sample", [])
        if snapshot_sample:
            first_row = snapshot_sample[0]
            for key in ["最新价", "成交量", "涨跌幅"]:
                if key in first_row:
                    evidence.append(f"{key}: {first_row[key]}")

    reason_map = {
        "trend_positive": "近三个月趋势和均线结构偏强，适合继续持有，但仍需控制仓位。",
        "trend_negative": "近三个月趋势偏弱，若已有持仓应谨慎，避免情绪化追高。",
        "trend_mixed": "趋势信号混合，当前更适合观察而不是激进加仓。",
    }

    return {
        "conclusion": conclusion,
        "confidence": confidence,
        "reason": reason_map.get(view, "趋势结构不明确。"),
        "evidence": evidence[:8],
    }


def _internal_resolve_a_stock_symbol(symbol_or_name: str) -> str | None:
    cleaned = str(symbol_or_name or "").strip()
    if not cleaned:
        return None

    if re.fullmatch(r"\d{6}", cleaned):
        return cleaned

    try:
        with _internal_without_proxy_env():
            lookup = ak.stock_info_a_code_name()
    except Exception:
        return None

    if lookup is None or getattr(lookup, "empty", True):
        return None

    code_column, name_column = _internal_lookup_columns(lookup)

    candidate_columns = [col for col in [code_column, name_column] if col is not None]
    if not candidate_columns:
        candidate_columns = list(getattr(lookup, "columns", []))[:2]

    for column in candidate_columns:
        series = lookup[column].astype(str)
        exact_matches = lookup[series == cleaned]
        if not exact_matches.empty and code_column is not None:
            resolved = str(exact_matches.iloc[0][code_column]).strip()
            if resolved:
                return resolved

        contains_matches = lookup[
            series.str.contains(cleaned, case=False, na=False, regex=False)
        ]
        if not contains_matches.empty and code_column is not None:
            resolved = str(contains_matches.iloc[0][code_column]).strip()
            if resolved:
                return resolved

    return None


def _internal_market_prefixed_symbol(symbol: str) -> str:
    cleaned = str(symbol or "").strip()
    if cleaned.startswith("6"):
        return f"sh{cleaned}"
    return f"sz{cleaned}"


def _internal_fetch_a_history_df(
    symbol: str,
    start_date: str,
    end_date: str,
    period: str = "daily",
    adjust: str = "",
):
    with _internal_without_proxy_env():
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as primary_exc:
            if period != "daily" or not hasattr(ak, "stock_zh_a_hist_tx"):
                raise primary_exc

            try:
                return ak.stock_zh_a_hist_tx(
                    symbol=_internal_market_prefixed_symbol(symbol),
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except Exception as fallback_exc:
                raise fallback_exc from primary_exc


@tool
def stock_a_history(
    symbol: str,
    start_date: str = "20250101",
    end_date: str = "20500101",
    period: str = "daily",
    adjust: str = "",
) -> str:
    """获取A股历史行情数据。"""

    try:
        start_value = _internal_normalize_date(start_date)
        end_value = _internal_normalize_date(end_date)
    except ValueError as exc:
        return json.dumps(
            {"tool": "stock_a_history", "error": str(exc)},
            ensure_ascii=False,
        )
    resolved_symbol = _internal_resolve_a_stock_symbol(symbol)
    if not resolved_symbol:
        return json.dumps(
            {
                "tool": "stock_a_history",
                "error": f"unable to resolve stock symbol or name: {symbol}",
            },
            ensure_ascii=False,
        )

    return _internal_safe_fetch(
        "stock_a_history",
        lambda: _internal_fetch_a_history_df(
            symbol=resolved_symbol,
            period=period,
            start_date=start_value,
            end_date=end_value,
            adjust=adjust,
        ),
    )


@tool
def stock_a_snapshot(symbol_or_name: str, history_days: int = 120, top_n: int = 10) -> str:
    """获取A股个股最新快照、最近历史和公司资料，优先按股票名称自动解析代码。"""

    resolved_symbol = _internal_resolve_a_stock_symbol(symbol_or_name)
    if not resolved_symbol:
        return json.dumps(
            {
                "tool": "stock_a_snapshot",
                "error": f"unable to resolve stock symbol or name: {symbol_or_name}",
            },
            ensure_ascii=False,
        )

    end_date = clock_now().strftime("%Y%m%d")
    start_date = (clock_now() - timedelta(days=max(1, int(history_days or 120)))).strftime("%Y%m%d")

    lookup_payload = None
    try:
        with _internal_without_proxy_env():
            lookup_df = ak.stock_info_a_code_name()
        lookup_payload = _internal_format_single_row(lookup_df, "stock_a_lookup", resolved_symbol, top_n=top_n)
    except Exception as exc:
        lookup_payload = {"tool": "stock_a_lookup", "error": str(exc)}

    spot_payload = None
    try:
        with _internal_without_proxy_env():
            spot_df = ak.stock_zh_a_spot_em()
        spot_payload = _internal_format_single_row(spot_df, "stock_a_spot", resolved_symbol, top_n=top_n)
    except Exception as exc:
        spot_payload = {"tool": "stock_a_spot", "error": str(exc)}

    history_payload = None
    history_metrics = {}
    try:
        history_df = _internal_fetch_a_history_df(
            symbol=resolved_symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
        history_payload = _internal_dataframe_payload("stock_a_history", history_df, keyword=resolved_symbol, top_n=top_n)
        history_metrics = _internal_history_metrics(history_df)
    except Exception as exc:
        history_payload = json.dumps({"tool": "stock_a_history", "error": str(exc)}, ensure_ascii=False)
        history_metrics = {"error": str(exc)}

    profile_payload = None
    try:
        with _internal_without_proxy_env():
            profile_df = ak.stock_individual_info_em(symbol=resolved_symbol)
        profile_payload = _internal_dataframe_payload("stock_a_individual_info", profile_df, keyword=resolved_symbol, top_n=top_n)
    except Exception as exc:
        profile_payload = json.dumps({"tool": "stock_a_individual_info", "error": str(exc)}, ensure_ascii=False)

    resolved_name = None
    try:
        with _internal_without_proxy_env():
            lookup_df = ak.stock_info_a_code_name()
        if lookup_df is not None and not getattr(lookup_df, "empty", True):
            code_col, name_col = _internal_lookup_columns(lookup_df)
            if code_col is not None and name_col is not None:
                match = lookup_df[lookup_df[code_col].astype(str) == resolved_symbol]
                if not match.empty:
                    resolved_name = str(match.iloc[0][name_col])
    except Exception:
        resolved_name = None

    return json.dumps(
        {
            "tool": "stock_a_snapshot",
            "input": symbol_or_name,
            "resolved_symbol": resolved_symbol,
            "resolved_name": resolved_name,
            "history_window_days": int(history_days or 120),
            "requested_as_of": end_date,
            "as_of": history_metrics.get("latest_trade_date") or end_date,
            "data_lag_days": history_metrics.get("data_lag_days"),
            "is_stale": history_metrics.get("is_stale"),
            "lookup": lookup_payload,
            "spot": spot_payload,
            "history": json.loads(history_payload) if isinstance(history_payload, str) else history_payload,
            "metrics": history_metrics,
            "profile": json.loads(profile_payload) if isinstance(profile_payload, str) else profile_payload,
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def stock_a_hold_analysis(symbol_or_name: str, history_days: int = 90) -> str:
    """基于最新个股快照和近三个月历史数据，给出是否适合持有的规则化判断。"""

    resolved_symbol = _internal_resolve_a_stock_symbol(symbol_or_name)
    if not resolved_symbol:
        return json.dumps(
            {
                "tool": "stock_a_hold_analysis",
                "error": f"unable to resolve stock symbol or name: {symbol_or_name}",
            },
            ensure_ascii=False,
        )

    end_date = clock_now().strftime("%Y%m%d")
    start_date = (clock_now() - timedelta(days=max(1, int(history_days or 90)))).strftime("%Y%m%d")

    resolved_name = None
    try:
        with _internal_without_proxy_env():
            lookup_df = ak.stock_info_a_code_name()
        if lookup_df is not None and not getattr(lookup_df, "empty", True):
            code_col, name_col = _internal_lookup_columns(lookup_df)
            if code_col is not None and name_col is not None:
                match = lookup_df[lookup_df[code_col].astype(str) == resolved_symbol]
                if not match.empty:
                    resolved_name = str(match.iloc[0][name_col])
    except Exception:
        resolved_name = None

    try:
        with _internal_without_proxy_env():
            snapshot_df = ak.stock_zh_a_spot_em()
        snapshot_payload = _internal_format_single_row(snapshot_df, "stock_a_spot", resolved_symbol, top_n=10)
    except Exception as exc:
        snapshot_payload = {"tool": "stock_a_spot", "error": str(exc)}

    try:
        history_df = _internal_fetch_a_history_df(
            symbol=resolved_symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
        history_payload = _internal_dataframe_payload("stock_a_history", history_df, keyword=None, top_n=10)
        history_metrics = _internal_history_metrics(history_df)
    except Exception as exc:
        history_payload = json.dumps({"tool": "stock_a_history", "error": str(exc)}, ensure_ascii=False)
        history_metrics = {"error": str(exc)}

    try:
        with _internal_without_proxy_env():
            info_df = ak.stock_individual_info_em(symbol=resolved_symbol)
        info_payload = _internal_dataframe_payload("stock_a_individual_info", info_df, keyword=None, top_n=20)
    except Exception as exc:
        info_payload = json.dumps({"tool": "stock_a_individual_info", "error": str(exc)}, ensure_ascii=False)

    view = history_metrics.get("view", "trend_mixed") if isinstance(history_metrics, dict) else "trend_mixed"
    hold_summary = _internal_build_hold_conclusion(history_metrics, json.loads(info_payload) if isinstance(info_payload, str) else info_payload, snapshot_payload)

    return json.dumps(
        {
            "tool": "stock_a_hold_analysis",
            "input": symbol_or_name,
            "resolved_symbol": resolved_symbol,
            "resolved_name": resolved_name,
            "requested_as_of": end_date,
            "as_of": history_metrics.get("latest_trade_date") or end_date,
            "data_lag_days": history_metrics.get("data_lag_days"),
            "is_stale": history_metrics.get("is_stale"),
            "history_window_days": int(history_days or 90),
            "hold_view": hold_summary.get("conclusion"),
            "hold_reason": hold_summary.get("reason"),
            "confidence": hold_summary.get("confidence"),
            "evidence": hold_summary.get("evidence", []),
            "snapshot": snapshot_payload,
            "history": json.loads(history_payload) if isinstance(history_payload, str) else history_payload,
            "metrics": history_metrics,
            "profile": json.loads(info_payload) if isinstance(info_payload, str) else info_payload,
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def stock_a_individual_info(symbol_or_name: str) -> str:
    """获取A股个股基础信息，优先按股票名称自动解析代码后查询东财个股信息。"""

    resolved_symbol = _internal_resolve_a_stock_symbol(symbol_or_name)
    if not resolved_symbol:
        return json.dumps(
            {
                "tool": "stock_a_individual_info",
                "error": f"unable to resolve stock symbol or name: {symbol_or_name}",
            },
            ensure_ascii=False,
        )

    return _internal_safe_fetch(
        "stock_a_individual_info",
        lambda: ak.stock_individual_info_em(symbol=resolved_symbol),
        keyword=None,
        top_n=20,
    )


@tool
def stock_a_conclusion(symbol_or_name: str, history_days: int = 90) -> str:
    """输出适合单只股票持有判断的结论前置摘要，包含结论、依据和关键数据。"""

    analysis_payload = json.loads(stock_a_hold_analysis.invoke({"symbol_or_name": symbol_or_name, "history_days": history_days}))
    if analysis_payload.get("error"):
        return json.dumps(analysis_payload, ensure_ascii=False)

    return json.dumps(
        {
            "tool": "stock_a_conclusion",
            "input": symbol_or_name,
            "resolved_symbol": analysis_payload.get("resolved_symbol"),
            "resolved_name": analysis_payload.get("resolved_name"),
            "as_of": analysis_payload.get("as_of"),
            "conclusion": analysis_payload.get("hold_view"),
            "confidence": analysis_payload.get("confidence"),
            "reason": analysis_payload.get("hold_reason"),
            "evidence": analysis_payload.get("evidence", []),
            "metrics": analysis_payload.get("metrics", {}),
        },
        ensure_ascii=False,
        default=str,
    )


def _internal_resolved_symbol_payload(tool_name: str, symbol_or_name: str) -> tuple[str | None, dict | None]:
    resolved_symbol = _internal_resolve_a_stock_symbol(symbol_or_name)
    if not resolved_symbol:
        return None, {
            "tool": tool_name,
            "error": f"unable to resolve stock symbol or name: {symbol_or_name}",
            "fetch_time": clock_now().isoformat(),
        }
    return resolved_symbol, None


def _internal_json_payload(tool_name: str, symbol_or_name: str, fetch_fn, top_n: int = 10) -> str:
    resolved_symbol, error_payload = _internal_resolved_symbol_payload(tool_name, symbol_or_name)
    if error_payload:
        return json.dumps(error_payload, ensure_ascii=False)
    try:
        with _internal_without_proxy_env():
            df = fetch_fn(resolved_symbol)
        payload = json.loads(_internal_dataframe_payload(tool_name, df, keyword=None, top_n=top_n))
        payload.update(
            {
                "input": symbol_or_name,
                "resolved_symbol": resolved_symbol,
                "fetch_time": clock_now().isoformat(),
                "data_as_of": _internal_infer_payload_date(payload),
            }
        )
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps(
            {
                "tool": tool_name,
                "input": symbol_or_name,
                "resolved_symbol": resolved_symbol,
                "error": str(exc),
                "fetch_time": clock_now().isoformat(),
            },
            ensure_ascii=False,
        )


def _internal_infer_payload_date(payload: dict) -> str | None:
    candidates = ["日期", "公告日期", "报告期", "报告日期", "评级日期", "发布时间", "时间"]
    for row in payload.get("sample", []) or []:
        if not isinstance(row, dict):
            continue
        for key in candidates:
            if row.get(key):
                return str(row[key])
    return None


@tool
def stock_a_valuation(symbol_or_name: str, top_n: int = 10) -> str:
    """获取A股估值数据，优先使用百度股市通估值接口。"""

    def _internal_fetch(symbol: str):
        if hasattr(ak, "stock_zh_valuation_baidu"):
            return ak.stock_zh_valuation_baidu(symbol=symbol)
        if hasattr(ak, "stock_value_em"):
            return ak.stock_value_em(symbol=symbol)
        raise AttributeError("no valuation API available")

    return _internal_json_payload("stock_a_valuation", symbol_or_name, _internal_fetch, top_n=top_n)


@tool
def stock_a_financial_indicators(symbol_or_name: str, top_n: int = 12) -> str:
    """获取A股财务分析主要指标。"""

    def _internal_fetch(symbol: str):
        if hasattr(ak, "stock_financial_analysis_indicator_em"):
            return ak.stock_financial_analysis_indicator_em(symbol=symbol)
        if hasattr(ak, "stock_financial_abstract"):
            return ak.stock_financial_abstract(symbol=symbol)
        raise AttributeError("no financial indicator API available")

    return _internal_json_payload("stock_a_financial_indicators", symbol_or_name, _internal_fetch, top_n=top_n)


@tool
def stock_a_fund_flow(symbol_or_name: str, top_n: int = 10) -> str:
    """获取A股个股资金流数据。"""

    def _internal_fetch(symbol: str):
        if hasattr(ak, "stock_individual_fund_flow"):
            return ak.stock_individual_fund_flow(stock=symbol)
        if hasattr(ak, "stock_fund_flow_individual"):
            return ak.stock_fund_flow_individual(symbol=symbol)
        raise AttributeError("no fund flow API available")

    return _internal_json_payload("stock_a_fund_flow", symbol_or_name, _internal_fetch, top_n=top_n)


@tool
def stock_a_research_report(symbol_or_name: str, top_n: int = 10) -> str:
    """获取A股个股研报/评级相关数据。"""

    def _internal_fetch(symbol: str):
        if hasattr(ak, "stock_research_report_em"):
            return ak.stock_research_report_em(symbol=symbol)
        if hasattr(ak, "stock_rank_forecast_cninfo"):
            return ak.stock_rank_forecast_cninfo(symbol=symbol)
        raise AttributeError("no research report API available")

    return _internal_json_payload("stock_a_research_report", symbol_or_name, _internal_fetch, top_n=top_n)


@tool
def stock_a_notice(symbol_or_name: str, top_n: int = 10) -> str:
    """获取A股个股公告数据。"""

    def _internal_fetch(symbol: str):
        if hasattr(ak, "stock_individual_notice_report"):
            return ak.stock_individual_notice_report(symbol=symbol)
        if hasattr(ak, "stock_notice_report"):
            return ak.stock_notice_report(symbol=symbol)
        raise AttributeError("no notice API available")

    return _internal_json_payload("stock_a_notice", symbol_or_name, _internal_fetch, top_n=top_n)


@tool
def stock_a_lookup(keyword: str, top_n: int = 20) -> str:
    """查询A股股票代码与名称映射。"""

    def _internal_fetch():
        lookup = ak.stock_info_a_code_name()
        if lookup is None:
            return lookup

        if getattr(lookup, "empty", True):
            return lookup

        keyword_text = str(keyword).strip()
        if not keyword_text:
            return lookup.head(max(1, min(int(top_n or 20), 50)))

        mask = lookup.astype(str).apply(
            lambda row: row.str.contains(
                keyword_text,
                case=False,
                na=False,
                regex=False,
            ).any(),
            axis=1,
        )
        filtered = lookup[mask]
        if getattr(filtered, "empty", True):
            return lookup.iloc[0:0]
        return filtered.head(max(1, min(int(top_n or 20), 50)))

    return _internal_safe_fetch(
        "stock_a_lookup",
        _internal_fetch,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def stock_a_spot(keyword: str | None = None, top_n: int = 20) -> str:
    """获取A股实时行情数据。"""

    return _internal_safe_fetch(
        "stock_a_spot",
        ak.stock_zh_a_spot_em,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def market_summary(exchange: str = "sse", date: str | None = None) -> str:
    """获取上交所或深交所市场概况。"""

    normalized = str(exchange).strip().lower()

    if normalized in {"sse", "sh", "shanghai", "上交所", "沪市"}:
        return _internal_safe_fetch("market_summary_sse", ak.stock_sse_summary)

    try:
        summary_date = _internal_normalize_date(date)
    except ValueError as exc:
        return json.dumps(
            {"tool": "market_summary_szse", "error": str(exc)},
            ensure_ascii=False,
        )
    return _internal_safe_fetch(
        "market_summary_szse",
        lambda: ak.stock_szse_summary(date=summary_date),
    )


@tool
def index_spot(symbol: str = "沪深重要指数", keyword: str | None = None, top_n: int = 20) -> str:
    """获取A股指数实时行情数据。"""

    return _internal_safe_fetch(
        "index_spot",
        lambda: ak.stock_zh_index_spot_em(symbol=symbol),
        keyword=keyword,
        top_n=top_n,
    )


@tool
def index_history(
    symbol: str,
    period: str = "daily",
    start_date: str = "20250101",
    end_date: str = "20500101",
) -> str:
    """获取A股指数历史行情数据。"""

    try:
        start_value = _internal_normalize_date(start_date)
        end_value = _internal_normalize_date(end_date)
    except ValueError as exc:
        return json.dumps(
            {"tool": "index_history", "error": str(exc)},
            ensure_ascii=False,
        )
    return _internal_safe_fetch(
        "index_history",
        lambda: ak.index_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_value,
            end_date=end_value,
        ),
    )


@tool
def index_minute(
    symbol: str,
    period: str = "1",
    start_date: str = "1979-09-01 09:32:00",
    end_date: str = "2222-01-01 09:32:00",
) -> str:
    """获取A股指数分时行情数据。"""

    return _internal_safe_fetch(
        "index_minute",
        lambda: ak.index_zh_a_hist_min_em(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
        ),
    )


@tool
def index_classic_spot(keyword: str | None = None, top_n: int = 30) -> str:
    """获取常见A股经典指数实时行情，并默认筛选常见基金/指数名称。"""

    def _internal_fetch():
        with _internal_without_proxy_env():
            return _internal_name_filter(
                ak.stock_zh_index_spot_sina(),
                _CLASSIC_INDEX_NAMES,
                keyword=keyword,
                top_n=top_n,
            )

    return _internal_safe_fetch(
        "index_classic_spot",
        _internal_fetch,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def hk_index_spot(keyword: str | None = None, top_n: int = 10) -> str:
    """获取港股指数实时行情，并默认筛选经典港股指数。"""

    def _internal_fetch():
        with _internal_without_proxy_env():
            return _internal_name_filter(
                ak.stock_hk_index_spot_em(),
                _CLASSIC_HK_INDEX_NAMES,
                keyword=keyword,
                top_n=top_n,
            )

    return _internal_safe_fetch(
        "hk_index_spot",
        _internal_fetch,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def hk_index_history(symbol: str) -> str:
    """获取港股指数历史行情数据。"""

    return _internal_safe_fetch(
        "hk_index_history",
        lambda: ak.stock_hk_index_daily_em(symbol=symbol),
    )


@tool
def fund_etf_spot(keyword: str | None = None, top_n: int = 20) -> str:
    """获取ETF基金实时行情数据。"""

    return _internal_safe_fetch(
        "fund_etf_spot",
        ak.fund_etf_spot_em,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def fund_open_daily(keyword: str | None = None, top_n: int = 20) -> str:
    """获取开放式基金日行情数据。"""

    return _internal_safe_fetch(
        "fund_open_daily",
        ak.fund_open_fund_daily_em,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def fund_etf_daily(keyword: str | None = None, top_n: int = 20) -> str:
    """获取ETF基金日行情数据。"""

    return _internal_safe_fetch(
        "fund_etf_daily",
        ak.fund_etf_fund_daily_em,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def fund_money_daily(keyword: str | None = None, top_n: int = 20) -> str:
    """获取货币基金日行情数据。"""

    return _internal_safe_fetch(
        "fund_money_daily",
        ak.fund_money_fund_daily_em,
        keyword=keyword,
        top_n=top_n,
    )


@tool
def fund_index_spot(keyword: str | None = None, top_n: int = 30) -> str:
    """查询基金/指数实时行情数据，并默认筛选常见基金指数名称。"""

    def _internal_fetch():
        with _internal_without_proxy_env():
            return _internal_name_filter(
                ak.stock_zh_index_spot_sina(),
                _CLASSIC_INDEX_NAMES,
                keyword=keyword,
                top_n=top_n,
            )

    return _internal_safe_fetch(
        "fund_index_spot",
        _internal_fetch,
        keyword=keyword,
        top_n=top_n,
    )


def get_akshare_tools():
    return [
        stock_a_history,
        stock_a_snapshot,
        stock_a_hold_analysis,
        stock_a_conclusion,
        stock_a_individual_info,
        stock_a_valuation,
        stock_a_financial_indicators,
        stock_a_fund_flow,
        stock_a_research_report,
        stock_a_notice,
        stock_a_lookup,
        stock_a_spot,
        market_summary,
        index_spot,
        index_history,
        index_minute,
        index_classic_spot,
        hk_index_spot,
        hk_index_history,
        fund_etf_spot,
        fund_open_daily,
        fund_etf_daily,
        fund_money_daily,
        fund_index_spot,
    ]
