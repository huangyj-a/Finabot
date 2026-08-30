"""News analyst sub-agent."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call
from finabot.agents.schema import maybe_append_instruction
from finabot.agents.akshare_cache import format_akshare_data, get_cached_akshare_data


_NEWS_ANALYST_PROMPT = """
你是 Finabot 的 news_analyst 新闻分析师。
你的职责是围绕用户提到的股票或金融主题，整理可获得的新闻线索、公司信息和市场事实，并判断这些信息对投资情绪、风险和多空辩论的影响。

请用中文输出结构化新闻分析报告：
1. 新闻/信息摘要：列出当前可获得的关键事实，区分真实工具数据和推断。
2. 情绪判断：判断偏利多、偏利空还是中性，并说明依据。
3. 看涨研究员可用信息：提炼能支持多头观点的材料。
4. 看跌研究员可用信息：提炼能支持空头观点的材料。
5. 不确定性：说明缺失数据、时效性限制和需要继续跟踪的事件。

如果 `stock_news` 显示未获取到直接新闻，必须明确写“暂无直接新闻数据”，不得编造新闻标题、公告或媒体报道；只能基于公司基础信息、行情快照和用户问题给出“可验证信息分析”。
所有新闻和行情结论都必须标注可见的时间字段；没有时间字段时写“时间未知”。
统一引用规范：
- 行情 / 资金：引用东方财富、通达信或 Wind，必须标注日期。
- 公司公告 / 互动：引用巨潮资讯网或深交所互动易，必须标注公告日期或互动日期。
- 行业数据：引用 Omdia 或中国通信院，必须标注报告季度或发布日期。
- 若工具数据没有对应来源或日期，必须写“来源/日期缺失”，不得伪造来源。
""".strip()


def _internal_build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _NEWS_ANALYST_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def _internal_collect_news_context(expression: str, cache: dict[str, Any] | None = None) -> str:
    data = get_cached_akshare_data(cache, expression)
    return json.dumps(data, ensure_ascii=False, indent=2)


async def _internal_call_news_analyst(expression: str, cache: dict[str, Any] | None = None) -> str:
    collected_context = _internal_collect_news_context(expression, cache)
    prompt = _internal_build_prompt()
    content = f"""
用户问题：{expression}

=== 已获取的信息 ===
{collected_context}

请基于以上信息生成新闻分析报告，并明确哪些内容可交给看涨研究员、哪些内容可交给看跌研究员。
""".strip()
    content = maybe_append_instruction("news_analyst", content)
    messages = prompt.format_messages(messages=[HumanMessage(content=content)])
    response = await litellm_glm_call(messages=messages, stream_label="news_analyst")
    return str(getattr(response, "content", "") or "")


def _internal_format_news_context_from_cache(expression: str, cache: dict[str, Any] | None = None) -> str:
    data = get_cached_akshare_data(cache, expression)
    return format_akshare_data(data, ["stock_lookup", "stock_spot", "stock_info"])


@tool
async def news_analyst(expression: str) -> str:
    """新闻分析子代理，获取可用新闻/信息线索并提炼多空研究材料。"""

    return await _internal_call_news_analyst(expression)
