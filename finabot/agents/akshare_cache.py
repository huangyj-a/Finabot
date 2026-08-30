"""Per-run AKShare data cache helpers."""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

from finabot.utils.clock import now as clock_now


def _internal_extract_stock_query(expression: str) -> str:
    text = str(expression or "").strip()
    code_match = re.search(r"\b(?:sh|sz)?(\d{6})\b", text, re.IGNORECASE)
    if code_match:
        return code_match.group(1)
    cleaned = re.sub(r"(帮我|请|分析|未来|三个月|是否|适合|持有|买入|卖出|继续|拿着|一下|关于|股票|的|，|。|\?|？|:|：)", " ", text)
    tokens = [token for token in re.split(r"\s+", cleaned.strip()) if token]
    return tokens[-1] if tokens else text


def get_cached_akshare_data(cache: dict[str, Any] | None, expression: str) -> dict[str, str]:
    """Fetch reusable AKShare payloads once per graph run."""

    if cache is None:
        cache = {}
    query = _internal_extract_stock_query(expression)
    if query in cache:
        return cache[query]
    for cached_payload in cache.values():
        if isinstance(cached_payload, dict) and cached_payload.get("resolved_symbol") == query:
            cache[query] = cached_payload
            return cached_payload

    from finabot.tools.akshare_tools import (
        stock_a_conclusion,
        stock_a_financial_indicators,
        stock_a_fund_flow,
        stock_a_individual_info,
        stock_a_lookup,
        stock_a_notice,
        stock_a_research_report,
        stock_a_snapshot,
        stock_a_spot,
        stock_a_valuation,
    )
    from finabot.tools.news_tools import get_stock_news_unified

    payload: dict[str, str] = {"query": expression, "extracted_query": query, "fetch_time": clock_now().isoformat()}
    target = query
    try:
        lookup = stock_a_lookup.invoke({"keyword": query, "top_n": 5})
        payload["stock_lookup"] = str(lookup)
        lookup_payload = json.loads(str(lookup))
        candidates = lookup_payload.get("sample") or []
        if candidates and isinstance(candidates[0], dict):
            first_candidate = candidates[0]
            code = (
                first_candidate.get("代码")
                or first_candidate.get("证券代码")
                or first_candidate.get("code")
            )
            if code:
                target = str(code).strip()
                payload["resolved_symbol"] = target
                payload["resolved_name"] = str(
                    first_candidate.get("名称")
                    or first_candidate.get("证券简称")
                    or first_candidate.get("name")
                    or ""
                )
    except Exception as exc:
        payload["stock_lookup_error"] = str(exc)

    # 10 个行情字段之间互不依赖，并行拉取以缩短整体耗时
    # （线程池而非 asyncio.gather：akshare 的同步 HTTP 调用在并发会话下本就安全，
    #  见 _internal_without_proxy_env 的历史说明）。
    field_specs = [
        ("stock_spot", stock_a_spot, {"keyword": target, "top_n": 5}),
        ("stock_info", stock_a_individual_info, {"symbol_or_name": target}),
        ("stock_snapshot", stock_a_snapshot, {"symbol_or_name": target, "history_days": 90, "top_n": 8}),
        ("stock_conclusion", stock_a_conclusion, {"symbol_or_name": target, "history_days": 90}),
        ("stock_valuation", stock_a_valuation, {"symbol_or_name": target, "top_n": 10}),
        ("stock_financial_indicators", stock_a_financial_indicators, {"symbol_or_name": target, "top_n": 12}),
        ("stock_fund_flow", stock_a_fund_flow, {"symbol_or_name": target, "top_n": 10}),
        ("stock_research_report", stock_a_research_report, {"symbol_or_name": target, "top_n": 10}),
        ("stock_notice", stock_a_notice, {"symbol_or_name": target, "top_n": 10}),
        ("stock_news", get_stock_news_unified, {"stock_code": target, "max_news": 10}),
    ]
    fetched: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(field_specs), 8)) as executor:
        futures = {
            executor.submit(func.invoke, kwargs): field
            for field, func, kwargs in field_specs
        }
        for future in concurrent.futures.as_completed(futures):
            field = futures[future]
            try:
                fetched[field] = future.result()
            except Exception as exc:
                fetched[field] = exc

    for field, _func, _kwargs in field_specs:
        outcome = fetched.get(field)
        if isinstance(outcome, Exception):
            payload[f"{field}_error"] = str(outcome)
        else:
            payload[field] = str(outcome)

    cache[query] = payload
    if target != query:
        cache[target] = payload
    return payload


def format_akshare_data(data: dict[str, str], fields: list[str] | None = None) -> str:
    selected_fields = fields or list(data.keys())
    sections = []
    for field in selected_fields:
        if field in data:
            sections.append(f"## {field}\n{data[field]}")
    return "\n\n".join(sections)
