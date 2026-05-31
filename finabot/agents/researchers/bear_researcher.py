"""Bearish investment debate researcher."""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call


_BEAR_RESEARCHER_PROMPT = """
你是一位看跌分析师，负责论证不投资或谨慎持有股票的理由。

你的目标是提出合理的论证，强调风险、挑战和负面指标。利用提供的研究和数据来突出潜在不利因素，并有效反驳看涨论点。

请用中文回答，重点关注以下几个方面：
- 风险和挑战：突出市场饱和、财务不稳定、估值过高或宏观经济威胁。
- 竞争劣势：强调市场地位较弱、创新下降、政策约束或竞争对手威胁。
- 负面指标：使用财务数据、市场趋势、价格表现或最近不利消息支持立场。
- 反驳看涨观点：用具体数据和合理推理揭露看涨论点的弱点或过度乐观假设。
- 参与讨论：以对话风格回应看涨研究员观点，而不仅仅是列举事实。

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
            ("system", _BEAR_RESEARCHER_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def _internal_format_expression(expression: str, debate_context: dict | None = None) -> str:
    debate_context = debate_context or {}
    history = debate_context.get("history", "") or "暂无"
    opponent_last = debate_context.get("last_bull_argument", "") or "暂无"
    news_report = debate_context.get("news_report", "") or "暂无新闻分析报告"
    return f"""
用户问题：{expression}

新闻分析师报告：
{news_report}

辩论对话历史：
{history}

最后的看涨论点：
{opponent_last}

请使用以上信息提供令人信服的看跌论点，反驳看涨声明，并展示投资该股票的风险和弱点。
""".strip()


async def _internal_call_bear_researcher(expression: str, debate_context: dict | None = None) -> str:
    prompt = _internal_build_prompt()
    content = _internal_format_expression(expression, debate_context)
    messages = prompt.format_messages(messages=[HumanMessage(content=content)])
    response = await litellm_glm_call(messages=messages)
    return str(getattr(response, "content", "") or "")


@tool
async def bear_researcher(expression: str) -> str:
    """看跌研究员，从空头视角论证投资风险并反驳看涨观点。"""

    return await _internal_call_bear_researcher(expression)
