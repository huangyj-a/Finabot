"""Bullish investment debate researcher."""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call


_BULL_RESEARCHER_PROMPT = """
你是一位看涨分析师，负责为股票投资建立强有力的论证。

你的任务是构建基于证据的强有力案例，强调增长潜力、竞争优势和积极的市场指标。利用提供的研究和数据来解决担忧并有效反驳看跌论点。

请用中文回答，重点关注以下几个方面：
- 增长潜力：突出公司的市场机会、收入预测和可扩展性。
- 竞争优势：强调独特产品、强势品牌、成本优势或主导市场地位。
- 积极指标：使用财务健康状况、行业趋势、价格表现和最新积极消息作为证据。
- 反驳看跌观点：用具体数据和合理推理回应风险担忧，说明为什么看涨观点更有说服力。
- 参与讨论：以对话风格回应看跌研究员观点，而不仅仅是列举数据。

如果缺少实时行情、基本面、新闻或情绪资料，请明确说明“暂无数据”，不要编造事实。
统一引用规范：
- 行情 / 资金：引用东方财富、通达信或 Wind，必须标注日期。
- 公司公告 / 互动：引用巨潮资讯网或深交所互动易，必须标注公告日期或互动日期。
- 行业数据：引用 Omdia 或中国通信院，必须标注报告季度或发布日期。
- 若工具数据没有对应来源或日期，必须写“来源/日期缺失”，不得伪造来源。
请确保所有回答都使用中文。
""".strip()


def _internal_build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _BULL_RESEARCHER_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def _internal_format_expression(expression: str, debate_context: dict | None = None) -> str:
    debate_context = debate_context or {}
    history = debate_context.get("history", "") or "暂无"
    opponent_last = debate_context.get("last_bear_argument", "") or "暂无"
    news_report = debate_context.get("news_report", "") or "暂无新闻分析报告"
    return f"""
用户问题：{expression}

新闻分析师报告：
{news_report}

辩论对话历史：
{history}

最后的看跌论点：
{opponent_last}

请使用以上信息提供令人信服的看涨论点，反驳看跌担忧，并展示看涨立场的优势。
""".strip()


async def _internal_call_bull_researcher(expression: str, debate_context: dict | None = None) -> str:
    prompt = _internal_build_prompt()
    content = _internal_format_expression(expression, debate_context)
    messages = prompt.format_messages(messages=[HumanMessage(content=content)])
    response = await litellm_glm_call(messages=messages)
    return str(getattr(response, "content", "") or "")


@tool
async def bull_researcher(expression: str) -> str:
    """看涨研究员，从多头视角论证股票投资机会并反驳看跌观点。"""

    return await _internal_call_bull_researcher(expression)
