import os
import json

from litellm import acompletion
from langchain_core.messages import BaseMessage, SystemMessage


def resolve_provider(provider: str | None) -> str:
    """Map legacy provider names to LiteLLM provider slugs."""
    normalized = (provider or "zai").strip().lower()
    if normalized in {"zhipu", "zhipuai", "glm"}:
        return "zai"
    return normalized


# 从环境变量读取供应商和模型
LLM_PROVIDER = resolve_provider(os.getenv("LLM_PROVIDER"))
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4")
LITELLM_MODEL = f"{LLM_PROVIDER}/{LLM_MODEL}"
API_KEY = os.getenv(f"{LLM_PROVIDER.upper()}_API_KEY") or os.getenv("ZHIPU_API_KEY")

# 系统提示词（上下文 Context 核心）
SYSTEM_PROMPT = """
你是 Finabot，一个智能理财助手。
请简洁、专业、准确地回答用户问题。
"""

def convert_messages(messages: list[BaseMessage]) -> list[dict]:
    dicts = []
    # 加入系统上下文
    dicts.append({"role": "system", "content": SYSTEM_PROMPT})

    # 加入历史消息
    for msg in messages:
        if msg.type == "human":
            dicts.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            dicts.append({"role": "assistant", "content": msg.content})
        elif msg.type == "tool":
            dicts.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            })
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = []
            for call in msg.tool_calls:
                if isinstance(call, dict):
                    tool_calls.append(call)
                    continue

                tool_calls.append({
                    "id": getattr(call, "id", None),
                    "type": "function",
                    "function": {
                        "name": getattr(call, "name", ""),
                        "arguments": json.dumps(getattr(call, "args", {}), ensure_ascii=False),
                    },
                })

            dicts.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": tool_calls
            })
    return dicts

async def litellm_glm_call(messages: list[BaseMessage], tools=None):
    messages_dict = convert_messages(messages)

    response = await acompletion(
        model=LITELLM_MODEL,          # LiteLLM 格式：zhipu/glm-4
        api_key=API_KEY,
        messages=messages_dict,
        tools=tools,
        temperature=0.1
    )

    msg = response.choices[0].message
    return msg