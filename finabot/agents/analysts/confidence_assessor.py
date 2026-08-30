"""Deterministic data-quality / confidence assessor for hold analysis.

This stage scans the prefetched AKShare payload and classifies how complete
and trustworthy the downstream analysis can be, BEFORE the summary manager
forms its conclusion. It is intentionally a pure function (no LLM call):
coverage and error accounting are algorithmic, so adding it costs ~0 latency
and avoids burning tokens on mechanical counting.
"""

from __future__ import annotations

import json
from typing import Any


# 流水线向 summary_manager 提供的结构化行情字段（与 hold_pipeline 预取字段一致）
_EXPECTED_FIELDS = (
    "stock_conclusion",
    "stock_valuation",
    "stock_financial_indicators",
    "stock_fund_flow",
    "stock_research_report",
    "stock_notice",
    "stock_info",
    "stock_snapshot",
    "stock_spot",
)

_NEWS_FIELD = "stock_news"


def _internal_assess_confidence(akshare_data: dict[str, Any] | None, expression: str) -> dict[str, Any]:
    """扫描缓存中的行情数据，产出覆盖度、缺失项与置信评级。"""
    if not isinstance(akshare_data, dict):
        return {
            "level": "低",
            "score": 0,
            "coverage": {"covered": [], "failed": []},
            "news_scope": None,
            "notes": "无行情数据",
        }

    covered: list[str] = []
    failed: list[dict[str, str]] = []

    for field in _EXPECTED_FIELDS:
        value = akshare_data.get(field)
        if not value or not str(value).strip():
            failed.append({"field": field, "error": "无返回"})
            continue
        try:
            parsed = json.loads(str(value))
        except (ValueError, TypeError):
            # 非 JSON 文本（简述/兜底字符串）按有内容计入
            covered.append(field)
            continue
        if isinstance(parsed, dict) and parsed.get("error"):
            failed.append({"field": field, "error": str(parsed["error"])[:200]})
        else:
            covered.append(field)

    news_scope = None
    news_raw = akshare_data.get(_NEWS_FIELD)
    if news_raw:
        try:
            news_parsed = json.loads(str(news_raw))
            if isinstance(news_parsed, dict):
                news_scope = news_parsed.get("news_scope")
        except (ValueError, TypeError):
            pass

    total = len(_EXPECTED_FIELDS)
    covered_n = len(covered)
    coverage_ratio = covered_n / total if total else 0

    if not failed and coverage_ratio >= 0.8:
        level, score = "高", 90
    elif coverage_ratio >= 0.5:
        level, score = "中", 70
    else:
        level, score = "低", 40
    # 多数行情工具系统性失败时再降级
    if len(failed) >= total * 0.5:
        level, score = "低", 30

    notes: list[str] = []
    if failed:
        notes.append("以下行情工具缺失或报错：" + "、".join(item["field"] for item in failed))
    if news_scope == "stock_direct":
        notes.append("有直接个股新闻数据")
    elif news_scope:
        notes.append("仅有市场通用新闻，缺直接个股新闻")
    if not notes:
        notes.append("行情数据完整")

    return {
        "level": level,
        "score": score,
        "coverage": {"covered": covered, "failed": failed},
        "news_scope": news_scope,
        "notes": "；".join(notes),
    }


def build_confidence_report(assessment: dict[str, Any]) -> str:
    """把置信评估转成 summary_manager 可直接拼接的文本段落。"""
    coverage = assessment.get("coverage", {})
    failed = coverage.get("failed", []) or []
    lines = [
        "### 数据质量与置信度",
        f"- 置信评级：{assessment.get('level', '未知')}（{assessment.get('score', 0)}/100）",
        f"- 行情覆盖：{len(coverage.get('covered', []))} 项有数据，{len(failed)} 项缺失/报错",
    ]
    if failed:
        lines.append(
            "- 缺失工具：" + "；".join(f"{item['field']}（{item['error']}）" for item in failed)
        )
    if assessment.get("notes"):
        lines.append(f"- 说明：{assessment['notes']}")
    return "\n".join(lines)
