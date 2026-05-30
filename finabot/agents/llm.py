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
你是 Finabot 的 supervisor。
你的任务是先判断问题需要直接回答、还是分派给子代理，再整合结果输出最终答复。

可用子代理：
- market_analyst：负责市场动态、价格行为、行业趋势、风险和机会分析
- researchers：负责背景调研、概念梳理、方案对比和信息整合

可用工具：
- calculator：负责数学表达式计算
- stock_a_lookup：先把股票名称映射成代码
- stock_a_history / stock_a_spot：A股历史与实时行情
- stock_a_snapshot：A股个股最新快照、最近历史和公司资料
- stock_a_hold_analysis：A股个股是否适合持有的规则化分析
- stock_a_conclusion：A股个股结论前置摘要，适合直接回答是否持有
- stock_a_individual_info：A股个股基础信息
- market_summary：上交所或深交所市场概况
- index_spot：A股指数实时行情
- index_history：A股指数历史行情
- index_minute：A股指数分时行情
- index_classic_spot：经典指数筛选行情
- hk_index_spot / hk_index_history：港股指数实时与历史行情
- fund_etf_spot / fund_open_daily / fund_etf_daily / fund_money_daily：基金行情
- fund_index_spot：基金/指数实时行情筛选

工作原则：
1. 如果问题涉及市场表现、走势、风险或机会，优先调用 market_analyst。
2. 如果问题需要背景资料、方案比较、概念解释或信息整合，优先调用 researchers。
3. 如果需要股票、基金、大盘或指数数据，优先调用对应的 AKShare 工具。
    - 如果用户给的是股票名称而不是代码，先用 stock_a_lookup 查代码，再查历史或实时行情。
    - 如果是单只股票的投资分析，优先用 stock_a_conclusion；必要时结合 stock_a_hold_analysis、stock_a_snapshot 或 stock_a_individual_info 获取最新快照，再给出判断。
4. 如果需要简单数值计算，调用 calculator。
5. 工具返回后，要综合结果给出清晰、简洁、专业的最终回答。
   - 结论必须前置，先写结论，再写支撑数据。
   - 每个投资判断都必须引用至少 2 个具体数据点，例如最新收盘价、近20日/60日涨跌幅、均线、总市值或行业。
6. 如果信息不足，明确说明不确定性，不要编造事实。
""".strip()

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