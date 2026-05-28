import json
import re

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call

from finabot.agents.llm import litellm_glm_call
from finabot.agents.state import AgentState
from finabot.tools.base import get_tools


tools = get_tools()


_TOOL_CALL_TEXT_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][\w\-]*)\s*(?:```(?:html)?\s*)?(?:<tool_call>)?.*?"
    r"<arg_key>\s*(?P<key>[^<]+?)\s*</arg_key>\s*"
    r"<arg_value>\s*(?P<value>.*?)\s*</arg_value>.*?(?:</tool_call>)?",
    re.IGNORECASE | re.DOTALL,
)


def normalize_tool_call(call):
    """Convert LiteLLM/OpenAI tool-call objects to the LangChain ToolCall shape."""
    if isinstance(call, dict):
        if "function" in call:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError:
                    arguments = {"expression": raw_arguments}
            else:
                arguments = raw_arguments or {}
            return tool_call(
                name=function.get("name", ""),
                args=arguments,
                id=call.get("id"),
            )

        raw_arguments = call.get("args", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                arguments = {"expression": raw_arguments}
        else:
            arguments = raw_arguments or {}

        return tool_call(
            name=call.get("name", ""),
            args=arguments,
            id=call.get("id"),
        )

    function = getattr(call, "function", None)
    raw_arguments = getattr(function, "arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = {"expression": raw_arguments}
    else:
        arguments = raw_arguments or {}

    return tool_call(
        name=getattr(function, "name", ""),
        args=arguments,
        id=getattr(call, "id", None),
    )


def extract_tool_calls_from_content(content: str):
    """Fallback parser for assistant text that contains serialized tool markup."""
    if not content:
        return []

    match = _TOOL_CALL_TEXT_PATTERN.search(content)
    if not match:
        return []

    tool_name = match.group("name").strip()
    arg_key = match.group("key").strip()
    arg_value = match.group("value").strip()
    if not tool_name or not arg_key:
        return []

    arguments = {arg_key: arg_value}
    return [tool_call(name=tool_name, args=arguments, id=None)]


def format_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        } for t in tools
    ]


async def call_llm_node(state: AgentState):
    msg = await litellm_glm_call(
        messages=state["messages"],
        tools=format_tools()
    )

    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    tool_calls = [normalize_tool_call(call) for call in raw_tool_calls]
    content = getattr(msg, "content", "") or ""

    if not tool_calls:
        tool_calls = extract_tool_calls_from_content(content)
        if tool_calls:
            content = ""

    ai_msg = AIMessage(
        content=content,
        tool_calls=tool_calls
    )
    return {"messages": [ai_msg]}


async def call_tool_node(state: AgentState):
    last = state["messages"][-1]
    tool_results = []

    for call in last.tool_calls:
        normalized_call = normalize_tool_call(call)
        tool_name = normalized_call["name"]
        t = next(x for x in tools if x.name == tool_name)
        res = await t.ainvoke(normalized_call["args"])
        tool_results.append(
            ToolMessage(
                content=str(res),
                tool_call_id=normalized_call["id"]
            )
        )
    return {"messages": tool_results}


def should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool"
    return "end"
