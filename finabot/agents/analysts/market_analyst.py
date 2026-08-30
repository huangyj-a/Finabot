"""Market analyst sub-agent."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call
from finabot.agents.schema import maybe_append_instruction


_MARKET_ANALYST_PROMPT = """
你是 Finabot 的 market_analyst 子代理。
你的职责是分析市场动态、价格行为、行业趋势、风险和机会。
回答时保持简洁、专业、结构清晰。
如果信息不足，明确说明假设，不要编造实时数据。

输出建议使用以下结构：
1. 市场判断
2. 关键依据
3. 风险与机会
4. 结论
""".strip()


def _internal_build_market_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _MARKET_ANALYST_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


async def _internal_call_market_analyst(expression: str) -> str:
    prompt = _internal_build_market_prompt()
    content = maybe_append_instruction("market_analyst", expression)
    messages = prompt.format_messages(messages=[HumanMessage(content=content)])
    response = await litellm_glm_call(messages=messages, stream_label="market_analyst")
    return str(getattr(response, "content", "") or "")


@tool
async def market_analyst(expression: str) -> str:
    """市场分析子代理，分析市场趋势、风险和机会。"""

    return await _internal_call_market_analyst(expression)