"""Hold analysis pipeline implemented as a compiled LangGraph subgraph.

Each analysis step (fetch → fundamental → news → bull/bear → summary) is its
own graph node. `bull` and `bear` fan out in parallel from `news` and converge
at `summary`, so the multi-agent debate runs as a real graph (not a single node
doing `asyncio.gather`). This keeps the structure ready for node-level streaming
and observability without changing the public `run_hold_analysis_pipeline` API.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from finabot.agents.akshare_cache import format_akshare_data, get_cached_akshare_data
from finabot.agents.analysts.confidence_assessor import (
    _internal_assess_confidence,
    build_confidence_report,
)
from finabot.agents.analysts.fundamental_analyst import _internal_call_fundamental_analyst
from finabot.agents.analysts.news_analyst import _internal_call_news_analyst
from finabot.agents.managers.manager import _internal_call_summary_manager
from finabot.agents.researchers import _internal_call_bear_researcher, _internal_call_bull_researcher

KEY_SECTIONS = [
    "stock_conclusion",
    "stock_valuation",
    "stock_financial_indicators",
    "stock_fund_flow",
    "stock_research_report",
    "stock_notice",
    "stock_info",
    "stock_snapshot",
    "stock_spot",
]


class _HoldPipelineState(TypedDict):
    expression: str
    akshare_cache: dict[str, Any]
    market_report: str
    memories: list
    fundamentals_report_raw: str
    confidence_report: str
    fundamentals_report: str
    news_report: str
    bull_report: str
    bear_report: str
    summary_report: str
    # 每个节点把自己的产出作为 AIMessage 追加进该通道，供父图 astream(subgraphs=True)
    # 实时透出"新闻完成 → 看涨 → 看跌 → 结论"的分步进度（不污染父图持久化状态）。
    messages: Annotated[list, operator.add]


async def _fetch_node(state: _HoldPipelineState) -> dict[str, str]:
    """预取 AKShare 行情（线程池 + 总超时），并扫描覆盖度产出置信评级。"""
    akshare_data = await asyncio.wait_for(
        asyncio.to_thread(get_cached_akshare_data, state["akshare_cache"], state["expression"]),
        timeout=120,
    )
    fundamentals_report_raw = format_akshare_data(akshare_data, KEY_SECTIONS)
    confidence = _internal_assess_confidence(akshare_data, state["expression"])
    confidence_report = build_confidence_report(confidence)
    return {
        "fundamentals_report_raw": fundamentals_report_raw,
        "confidence_report": confidence_report,
        "messages": [AIMessage(content=f"📊 已预取行情数据，置信评级：{confidence.get('level', '未知')}")],
    }


async def _fundamental_node(state: _HoldPipelineState) -> dict[str, str]:
    result = await _internal_call_fundamental_analyst(state["expression"], state["akshare_cache"])
    return {"fundamentals_report": result, "messages": [AIMessage(content=result or "")]}


async def _news_node(state: _HoldPipelineState) -> dict[str, str]:
    result = await _internal_call_news_analyst(state["expression"], state["akshare_cache"])
    return {"news_report": result, "messages": [AIMessage(content=result or "")]}


async def _bull_node(state: _HoldPipelineState) -> dict[str, str]:
    debate_context = {"news_report": state.get("news_report", "")}
    result = await _internal_call_bull_researcher(state["expression"], debate_context)
    return {"bull_report": result, "messages": [AIMessage(content=result or "")]}


async def _bear_node(state: _HoldPipelineState) -> dict[str, str]:
    debate_context = {"news_report": state.get("news_report", "")}
    result = await _internal_call_bear_researcher(state["expression"], debate_context)
    return {"bear_report": result, "messages": [AIMessage(content=result or "")]}


async def _summary_node(state: _HoldPipelineState) -> dict[str, str]:
    fundamentals = state.get("fundamentals_report") or state.get("fundamentals_report_raw") or ""
    summary_report = await _internal_call_summary_manager(
        state["expression"],
        {
            "market_report": state.get("market_report", ""),
            "news_report": state.get("news_report", ""),
            "bull_report": state.get("bull_report", ""),
            "bear_report": state.get("bear_report", ""),
            "fundamentals_report": fundamentals,
            "confidence_report": state.get("confidence_report", ""),
            "memories": state.get("memories", []),
            "akshare_cache": state["akshare_cache"],
        },
    )
    return {"summary_report": summary_report, "messages": [AIMessage(content=summary_report or "")]}


def build_hold_analysis_subgraph():
    """编译持有分析子图：fetch → fundamental → news → (bull ∥ bear) → summary。"""
    g = StateGraph(_HoldPipelineState)
    g.add_node("fetch", _fetch_node)
    g.add_node("fundamental", _fundamental_node)
    g.add_node("news", _news_node)
    g.add_node("bull", _bull_node)
    g.add_node("bear", _bear_node)
    g.add_node("summary", _summary_node)

    g.add_edge(START, "fetch")
    g.add_edge("fetch", "fundamental")
    g.add_edge("fundamental", "news")
    # 多空并行扇出：bull 与 bear 互不依赖，分别独立运行后汇聚到 summary
    g.add_edge("news", "bull")
    g.add_edge("news", "bear")
    g.add_edge("bull", "summary")
    g.add_edge("bear", "summary")
    g.add_edge("summary", END)
    return g.compile()


# 编译一次，复用；run_hold_analysis_pipeline 只负责注入输入并读取结果
_HOLD_SUBGRAPH = build_hold_analysis_subgraph()


async def run_hold_analysis_pipeline(
    expression: str,
    state_context: dict[str, Any] | None = None,
    debate_mode: bool = False,
) -> dict[str, str]:
    """Fundamental → news → bull+bear (parallel) → summary as a compiled subgraph.

    debate_mode=True 时额外返回 debate_report：把新闻、看涨、看跌与综合结论
    拼接成一份可分步展示的稿件，供 supervisor 直接转述给用户。
    """

    state_context = state_context or {}
    akshare_cache = state_context.setdefault("akshare_cache", {})
    initial: _HoldPipelineState = {
        "expression": expression,
        "akshare_cache": akshare_cache,
        "market_report": state_context.get("market_report", ""),
        "memories": state_context.get("memories", []),
        "fundamentals_report_raw": "",
        "confidence_report": "",
        "fundamentals_report": "",
        "news_report": "",
        "bull_report": "",
        "bear_report": "",
        "summary_report": "",
        "messages": [],
    }

    result = await _HOLD_SUBGRAPH.ainvoke(initial)

    final_fundamentals = result.get("fundamentals_report") or result.get("fundamentals_report_raw") or ""
    out: dict[str, str] = {
        "fundamentals_report": final_fundamentals,
        "news_report": result.get("news_report", ""),
        "bull_report": result.get("bull_report", ""),
        "bear_report": result.get("bear_report", ""),
        "confidence_report": result.get("confidence_report", ""),
        "summary_report": result.get("summary_report", ""),
    }
    if debate_mode:
        out["debate_report"] = _internal_build_debate_report(
            out["news_report"], out["bull_report"], out["bear_report"], out["summary_report"]
        )
    return out


def _internal_build_debate_report(
    news_report: str,
    bull_report: str,
    bear_report: str,
    summary_report: str,
) -> str:
    """把新闻、多空与结论拼成分步展示稿件。"""
    sections: list[str] = []
    if news_report:
        sections.append(f"### 新闻分析\n{news_report}")
    if bull_report:
        sections.append(f"### 看涨论点\n{bull_report}")
    if bear_report:
        sections.append(f"### 看跌论点\n{bear_report}")
    sections.append(f"### 综合结论\n{summary_report}")
    return "\n\n".join(sections)
