"""Single-node hold analysis pipeline."""

from __future__ import annotations

from typing import Any

from finabot.agents.akshare_cache import format_akshare_data, get_cached_akshare_data
from finabot.agents.analysts.news_analyst import _internal_call_news_analyst
from finabot.agents.managers.manager import _internal_call_summary_manager
from finabot.agents.researchers import _internal_call_bear_researcher, _internal_call_bull_researcher


async def run_hold_analysis_pipeline(
    expression: str,
    state_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Run news, bull, bear, and summary in one graph node with shared AKShare cache."""

    state_context = state_context or {}
    akshare_cache = state_context.setdefault("akshare_cache", {})
    akshare_data = get_cached_akshare_data(akshare_cache, expression)
    fundamentals_report = format_akshare_data(
        akshare_data,
        [
            "stock_conclusion",
            "stock_valuation",
            "stock_financial_indicators",
            "stock_fund_flow",
            "stock_research_report",
            "stock_notice",
            "stock_info",
            "stock_snapshot",
            "stock_spot",
        ],
    )

    news_report = await _internal_call_news_analyst(expression, akshare_cache)
    debate_context = {"news_report": news_report}
    bull_report = await _internal_call_bull_researcher(expression, debate_context)
    debate_context["last_bull_argument"] = bull_report
    bear_report = await _internal_call_bear_researcher(expression, debate_context)

    summary_report = await _internal_call_summary_manager(
        expression,
        {
            "market_report": state_context.get("market_report", ""),
            "news_report": news_report,
            "bull_report": bull_report,
            "bear_report": bear_report,
            "fundamentals_report": fundamentals_report,
            "memories": state_context.get("memories", []),
            "akshare_cache": akshare_cache,
        },
    )
    return {
        "fundamentals_report": fundamentals_report,
        "news_report": news_report,
        "bull_report": bull_report,
        "bear_report": bear_report,
        "summary_report": summary_report,
    }
