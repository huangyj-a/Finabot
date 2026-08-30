"""Summary manager agent."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from finabot.agents.llm import litellm_glm_call
from finabot.agents.akshare_cache import format_akshare_data, get_cached_akshare_data


_SUMMARY_MANAGER_PROMPT = """
你是 Finabot 的总结分析师（summary_manager）。
你的职责是整合市场分析、新闻分析、看涨研究、看跌研究、股票基本数据和用户记忆，输出可执行、结构稳定、数据优先的金融分析结论。

当用户询问金融、股票、基金、指数或持仓决策时，必须参考 docs/examples1.md 的密度与结构，采用以下标准格式：

结论前置：用一句话给出未来一段时间是否适合持有/买入/观望，以及核心原因。

### 一、核心判断（未来一段时间）
- 区间预判：给出合理价格/指数区间；缺数据时写“暂无数据”。
- 持仓建议：给出仓位、加减仓和风格适配。
- 关键时间点：列出财报、政策、新闻、行业催化或风险窗口。

### 二、看多逻辑（支撑持有）
- 至少 2-3 条，优先使用市场分析、新闻分析、看涨研究和基本数据。
- 每条尽量包含具体数据；缺数据时明确“暂无数据”。

### 三、看空/风险逻辑（制约追高）
- 至少 2-3 条，优先使用新闻风险、看跌研究、估值/趋势/基本面风险。
- 不要淡化不确定性。

### 四、持仓策略（可直接执行）
- 给出仓位、加仓、减仓、止盈止损和跟踪项。

### 五、总结
- 用一段话收束“是否适合、为什么、适合什么风格”。

要求：中文回答；结论前置；不得编造实时数据；必须区分工具数据、研究员观点和推断。
关键数据必须写明日期或时间来源，例如 `as_of`、`latest_trade_date`、`fetch_time`、新闻发布时间；缺少日期时必须写“时间未知”。
如果 `is_stale=true` 或 `data_lag_days>7`，必须降低结论置信度，并在风险段明确提示数据过时。
如果新闻数据为“无法获取”或 `has_direct_news=false`，不得写“最新新闻显示”，只能写“暂无直接新闻数据”。
最高风险回显：如果输入中包含“最高风险清单”，必须在“看空/风险逻辑”段逐条回显清单中的最高级别风险，不得无解释地省略、淡化或让最高风险消失。
估值、PE、PEG、机构评级、资金流、公告、研报等判断必须优先引用 AKShare 工具数据；如果对应工具没有返回数据，不得自行给出具体数值。
基本面交叉验证：技术面定位（均线/区间）之外，必须纳入估值维度（TTM PE/PB 及历史分位，作为下方安全垫/支撑解释）与财务维度（盈利/毛利/营收增速），用 stock_a_valuation、stock_a_financial_indicators 的数据交叉验证；批价（如飞天批价）、渠道库存、提价/出厂价上调预期、业绩（如 Q3 报表）等若无直接工具数据，须列为关键定性观察点并标注“数据缺失/来源未知”。
情景推演补全：乐观情景的触发条件须包含“提价/出厂价上调预期”；悲观情景的风险须包含“业绩不及预期（动销走弱、报表继续承压）”而不仅是大盘系统性回调。
时间轴：以工具返回的 latest_trade_date / as_of 为“当前”，推算“未来两个月”≈其后约 60 个交易日（约 2 个月），区间与情景的时间标签须与数据日期一致，避免把未来两个月错标成具体月份（如把 8 月底起的未来两个月标成 11–12 月）。

统一引用规范：
- 行情 / 资金：引用东方财富、通达信或 Wind，必须标注日期。
- 公司公告 / 互动：引用巨潮资讯网或深交所互动易，必须标注公告日期或互动日期。
- 行业数据：引用 Omdia 或中国通信院，必须标注报告季度或发布日期。
- 若工具数据没有对应来源或日期，必须写“来源/日期缺失”，不得伪造来源。
""".strip()


def _internal_build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SUMMARY_MANAGER_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def _internal_collect_fundamental_context(expression: str, cache: dict[str, Any] | None = None) -> str:
    data = get_cached_akshare_data(cache, expression)
    return format_akshare_data(
        data,
        [
            "stock_conclusion",
            "stock_valuation",
            "stock_financial_indicators",
            "stock_fund_flow",
            "stock_research_report",
            "stock_notice",
            "stock_info",
            "stock_snapshot",
            "stock_spot",
        ],
    )


def _internal_format_summary_input(expression: str, context: dict | None = None) -> str:
    context = context or {}
    market_report = context.get("market_report") or "暂无市场分析数据"
    news_report = context.get("news_report") or "暂无新闻分析数据"
    bull_report = context.get("bull_report") or context.get("last_bull_argument") or "暂无看涨研究数据"
    bear_report = context.get("bear_report") or context.get("last_bear_argument") or "暂无看跌研究数据"
    fundamentals_report = context.get("fundamentals_report") or _internal_collect_fundamental_context(
        expression,
        context.get("akshare_cache"),
    )
    memories = context.get("memories") or []
    confidence_report = context.get("confidence_report") or ""
    risk_flags = context.get("risk_flags") or []

    risk_section = ""
    if risk_flags:
        risk_section = (
            "=== 最高风险清单（必须在“看空/风险逻辑”段逐条回显，不得无解释消失）===\n"
            + "\n".join(f"- {flag}" for flag in risk_flags)
        )

    return f"""
用户问题：{expression}

=== 股票基本数据 ===
{fundamentals_report}

=== 市场分析数据 ===
{market_report}

=== 新闻分析数据 ===
{news_report}

=== 看涨研究数据 ===
{bull_report}

=== 看跌研究数据 ===
{bear_report}

=== 用户记忆 ===
{memories}

{risk_section + chr(10) if risk_section else ""}{confidence_report + chr(10) if confidence_report else ""}=== 回答格式 ===
最终回答必须严格采用以下六段式结构，且每一段都要有真实数据支撑：
    - 结论前置：先给出未来一段时间的持有判断，再说明一句原因。
    - 核心判断：给出未来一段时间的区间预判、仓位建议和关键时间点。
    - 看多逻辑：至少列出 2-3 条支撑持有的逻辑，每条都要带数据。
    - 看空 / 风险逻辑：至少列出 2-3 条风险，每条都要带数据。
    - 持仓策略：给出可以直接执行的仓位、加减仓、止盈止损和跟踪项。
    - 最后总结：用一段话收束“是否适合持有、为什么、适合什么风格”。
    - 如果缺少某一类数据，明确写“暂无数据”，不要补写臆测内容。
    - 不要只给短结论；单股持有类回答必须尽量贴近 `docs/examples1.md` 的分析密度与条理。
请整合以上信息，按标准格式输出最终分析。若某类数据缺失，写“暂无数据”，不要编造。
""".strip()


async def _internal_call_summary_manager(expression: str, context: dict | None = None) -> str:
    prompt = _internal_build_prompt()
    effective_context = context or {}
    if not effective_context.get("fundamentals_report"):
        # AKShare 抓取是同步网络 IO，放到线程池执行，避免阻塞事件循环。
        effective_context = dict(effective_context)
        effective_context["fundamentals_report"] = await asyncio.to_thread(
            _internal_collect_fundamental_context,
            expression,
            effective_context.get("akshare_cache"),
        )
    content = _internal_format_summary_input(expression, effective_context)
    messages = prompt.format_messages(messages=[HumanMessage(content=content)])
    response = await litellm_glm_call(messages=messages, memories=effective_context.get("memories"), stream_label="summary_manager")
    return str(getattr(response, "content", "") or "")
