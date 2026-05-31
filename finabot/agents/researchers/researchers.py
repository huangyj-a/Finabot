"""General research sub-agent."""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call


_RESEARCH_PROMPT = """
你是 Finabot 的 researchers 子代理。
你的职责是做背景调研、概念梳理、方案比较和信息整合。
回答时保持中文、专业、结构清晰；如果缺少实时数据，明确说明限制，不要编造事实。

输出建议使用以下结构：
1. 背景概览
2. 关键事实
3. 分歧与不确定性
4. 综合结论
""".strip()


def _internal_build_research_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _RESEARCH_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


async def _internal_call_researchers(expression: str) -> str:
    prompt = _internal_build_research_prompt()
    messages = prompt.format_messages(messages=[HumanMessage(content=expression)])
    response = await litellm_glm_call(messages=messages)
    return str(getattr(response, "content", "") or "")


@tool
async def researchers(expression: str) -> str:
    """研究子代理，进行背景调研、概念梳理和信息整合。"""

    return await _internal_call_researchers(expression)
