"""Research sub-agent."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call


_RESEARCH_PROMPT = """
你是 Finabot 的 researchers 子代理。
你的职责是做背景调研、概念梳理、方案对比和信息整合。
回答时基于已有信息推理，清楚标注不确定性。
如果需要外部资料才能确认，请直接说明。

输出建议使用以下结构：
1. 调研目标
2. 已知信息
3. 关键发现
4. 未知项与风险
5. 建议下一步
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