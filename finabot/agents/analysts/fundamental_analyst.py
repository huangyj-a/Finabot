"""Fundamental analysis sub-agent."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call
from finabot.agents.akshare_cache import format_akshare_data, get_cached_akshare_data


_FUNDAMENTAL_ANALYST_PROMPT = """
你是 Finabot 的 fundamental_analyst（基本面分析师）。
你的职责是解读 AKShare 返回的原始财务数据，生成一份简洁、结构化的投研简报，
供后续多空研究和最终总结环节使用。

输入数据包括：股票快照、历史行情指标（涨跌幅、均线、趋势判断）、估值数据、
财务指标、资金流向、研报摘要、公告、公司资料等。

请用中文按以下结构输出：

### 基本面概览
- 一句话概括公司当前经营状态和投资价值定位

### 财务健康
- 营收与利润趋势（增长率、毛利率变化）
- ROE、负债率等关键指标及趋势方向

### 估值判断
- 当前 PE/PB 所处历史分位
- PEG 及相对行业估值水平

### 技术面信号
- 均线系统状态（金叉/死叉/多头排列）
- 支撑位/阻力位
- 近期涨跌动量（return_20d、return_60d）

### 资金面观察
- 主力/散户资金流向趋势
- 北向资金动态
- 机构持仓变化（如有）

### 风险提示
- 列出数据中明确可见的 2-3 个风险点
- 如果数据不足，写"数据不足，暂无法判断"

要求：
- 每个判断必须引用具体数据（数字 + 日期），不得编造
- 不要重复原始 JSON 格式，用人类可读的分析语言
- 如果某个数据模块缺失（对应工具返回 error），在该节写"数据暂缺"
- 时间字段缺失时写"时间未知"
- 统一引用规范：
  - 行情/资金：引用东方财富、通达信或 Wind，必须标注日期
  - 公司公告/互动：引用巨潮资讯网或深交所互动易
  - 行业数据：引用 Omdia 或中国通信院
  - 缺少来源时写"来源/日期缺失"
""".strip()


def _internal_build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _FUNDAMENTAL_ANALYST_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


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


async def _internal_call_fundamental_analyst(
    expression: str,
    akshare_cache: dict[str, Any] | None = None,
) -> str:
    """解读原始 AKShare 数据为结构化投研简报。"""
    akshare_data = await asyncio.to_thread(
        get_cached_akshare_data, akshare_cache, expression
    )
    formatted_data = format_akshare_data(akshare_data, KEY_SECTIONS)

    prompt = _internal_build_prompt()
    content = (
        f"用户问题：{expression}\n\n"
        f"=== 原始财务数据 ===\n{formatted_data}\n\n"
        f"请基于以上数据生成结构化基本面分析简报。"
    )
    messages = prompt.format_messages(messages=[HumanMessage(content=content)])
    response = await litellm_glm_call(messages=messages, stream_label="fundamental_analyst")
    return str(getattr(response, "content", "") or "")


@tool
async def fundamental_analyst(expression: str) -> str:
    """基本面分析子代理，解读财务/估值/技术/资金数据为结构化投研简报。"""
    return await _internal_call_fundamental_analyst(expression)
