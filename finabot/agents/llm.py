import os

from litellm import acompletion
from langchain_core.messages import BaseMessage

from finabot.agents.context import ContextBuilder


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
- news_analyst：负责获取可用新闻/信息线索并生成新闻分析报告，供多空研究员使用
- researchers：负责背景调研、概念梳理、方案对比和信息整合
- bull_researcher：看涨研究员，负责从多头角度论证投资机会并反驳看跌观点
- bear_researcher：看跌研究员，负责从空头角度论证投资风险并反驳看涨观点
- summary_manager：总结分析师，负责整合市场、新闻、多空研究和基本数据，按标准格式输出最终金融分析
- hold_analysis_pipeline：单股持有分析流水线，共享AKShare缓存并一次性完成新闻、多空和总结分析

可用工具：
- calculator：负责数学表达式计算
- read_file：按需读取 `skills/` 技能文件或 `.finabot_context/` 压缩上下文落盘文件
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
2. 如果问题涉及公司新闻、公告、事件影响、舆情或新闻驱动的投资判断，优先调用 news_analyst。
3. 如果用户询问“某只股票未来一段时间是否适合持有/买入/继续拿着”，优先调用 hold_analysis_pipeline，一次性完成新闻、多空和总结分析，避免重复调用多个节点。
4. 如果用户明确要求分开展示多空辩论，才按顺序调用 news_analyst、bull_researcher、bear_researcher、summary_manager；不要并行调用。
5. 如果问题需要背景资料、方案比较、概念解释或信息整合，优先调用 researchers。
6. 如果需要股票、基金、大盘或指数数据，优先调用对应的 AKShare 工具。
    - 如果用户给的是股票名称而不是代码，先用 stock_a_lookup 查代码，再查历史或实时行情。
    - 如果是单只股票的投资分析，优先用 stock_a_conclusion；必要时结合 stock_a_hold_analysis、stock_a_snapshot 或 stock_a_individual_info 获取最新快照，再给出判断。
7. 如果需要简单数值计算，调用 calculator。
8. 如果系统提示中的“按需加载”技能摘要与当前任务相关，先用 read_file 读取对应路径，再应用技能内容。
9. 工具返回后，要综合结果给出清晰、简洁、专业的最终回答。
   - 结论必须前置，先写结论，再写支撑数据。
   - 每个投资判断都必须引用至少 2 个具体数据点，例如最新收盘价、近20日/60日涨跌幅、均线、总市值或行业。
   - 对金融投资类问题，优先让 summary_manager 生成最终回答，不要只拼接工具原文。
10. 如果信息不足，明确说明不确定性，不要编造事实。

统一引用规范：
- 行情 / 资金：引用东方财富、通达信或 Wind，必须标注日期。
- 公司公告 / 互动：引用巨潮资讯网或深交所互动易，必须标注公告日期或互动日期。
- 行业数据：引用 Omdia 或中国通信院，必须标注报告季度或发布日期。
- 若工具数据没有对应来源或日期，必须写“来源/日期缺失”，不得伪造来源。
""".strip()

def convert_messages(messages: list[BaseMessage], memories=None, compression_mode="auto") -> list[dict]:
    builder = ContextBuilder(SYSTEM_PROMPT)
    return builder.build_messages(messages, memories=memories, compression_mode=compression_mode)

async def litellm_glm_call(messages: list[BaseMessage], tools=None, memories=None):
    messages_dict = convert_messages(messages, memories=memories)
    try:
        response = await acompletion(
            model=LITELLM_MODEL,          # LiteLLM 格式：zhipu/glm-4
            api_key=API_KEY,
            messages=messages_dict,
            tools=tools,
            temperature=0.1
        )
    except Exception as exc:
        if not _internal_is_prompt_too_long(exc):
            raise
        response = await acompletion(
            model=LITELLM_MODEL,
            api_key=API_KEY,
            messages=convert_messages(messages, memories=memories, compression_mode="reactive"),
            tools=tools,
            temperature=0.1
        )

    msg = response.choices[0].message
    return msg


def _internal_is_prompt_too_long(exc: Exception) -> bool:
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status_code == 413 or "prompt_too_long" in text or "context length" in text
