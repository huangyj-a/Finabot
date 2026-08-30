import asyncio
import os
import random
import sys
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any


def _debug_timing(label: str) -> None:
    """受 FINABOT_DEBUG_TIMING 控制的阶段耗时日志（仅诊断用，默认关闭）。"""
    if os.getenv("FINABOT_DEBUG_TIMING"):
        print(f"[timing] {label}", file=sys.stderr, flush=True)

from litellm import acompletion
from langchain_core.messages import BaseMessage

from finabot.agents.context import ContextBuilder
from finabot.agents.streaming import get_token_sink, is_streamable_label
from finabot.agents.telemetry import LLMCallMetric, LLM_METRICS, utc_timestamp


def resolve_provider(provider: str | None) -> str:
    """Map legacy provider names to LiteLLM provider slugs."""
    normalized = (provider or "zai").strip().lower()
    if normalized in {"zhipu", "zhipuai", "glm"}:
        return "zai"
    return normalized


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    litellm_model: str
    api_key: str
    timeout_seconds: float
    api_base: str | None = None


def get_llm_settings() -> LLMSettings:
    """Resolve current LLM configuration after environment files are loaded."""

    provider = resolve_provider(os.getenv("LLM_PROVIDER"))
    model = os.getenv("LLM_MODEL", "glm-4").strip() or "glm-4"
    # 支持自定义 OpenAI 兼容端点：LLM_API_BASE（通用）优先，其次 {PROVIDER}_API_BASE。
    api_base = (
        os.getenv("LLM_API_BASE", "").strip()
        or os.getenv(f"{provider.upper()}_API_BASE", "").strip()
        or None
    )
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv(f"{provider.upper()}_API_KEY")
        or os.getenv("ZHIPU_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            f"Missing API key: set LLM_API_KEY or {provider.upper()}_API_KEY (fallback ZHIPU_API_KEY) before starting Finabot."
        )
    try:
        timeout_seconds = float(os.getenv("FINABOT_LLM_TIMEOUT_SECONDS", "90"))
    except ValueError as exc:
        raise RuntimeError("FINABOT_LLM_TIMEOUT_SECONDS must be a positive number.") from exc
    if timeout_seconds <= 0:
        raise RuntimeError("FINABOT_LLM_TIMEOUT_SECONDS must be a positive number.")
    return LLMSettings(
        provider=provider,
        model=model,
        litellm_model=f"{provider}/{model}",
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        api_base=api_base,
    )

# 系统提示词（上下文 Context 核心）
SYSTEM_PROMPT = """
你是 Finabot 的 supervisor。
你的任务是先判断问题需要直接回答、还是分派给子代理，再整合结果输出最终答复。

可用子代理：
- market_analyst：负责市场动态、价格行为、行业趋势、风险和机会分析
- fundamental_analyst：负责解读财务/估值/技术/资金数据，生成结构化基本面投研简报
- news_analyst：负责获取可用新闻/信息线索并生成新闻分析报告，供多空研究员使用
- researchers：负责背景调研、概念梳理、方案对比和信息整合
- hold_analysis_pipeline：单股持有分析流水线，共享AKShare缓存并一次性完成新闻、多空和总结分析（默认输出简洁结论）

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
2. 如果问题需要深入解读具体股票的财务数据、估值水平、技术指标或资金流向，优先调用 fundamental_analyst。
3. 如果问题涉及公司新闻、公告、事件影响、舆情或新闻驱动的投资判断，优先调用 news_analyst。
4. 如果用户询问“某只股票未来一段时间是否适合持有/买入/继续拿着”，优先调用 hold_analysis_pipeline，一次性完成新闻、多空和总结分析，避免重复调用多个节点。
5. 如果用户明确要求分步骤展示多空辩论过程，调用 hold_analysis_pipeline 时传入 debate_mode=true；否则默认不传，直接给简洁结论。不要并行调用多个子代理。
6. 如果问题需要背景资料、方案比较、概念解释或信息整合，优先调用 researchers。
7. 如果需要股票、基金、大盘或指数数据，优先调用对应的 AKShare 工具。
    - 如果用户给的是股票名称而不是代码，先用 stock_a_lookup 查代码，再查历史或实时行情。
    - 如果是单只股票的投资分析，优先用 stock_a_conclusion；必要时结合 stock_a_hold_analysis、stock_a_snapshot 或 stock_a_individual_info 获取最新快照，再给出判断。
8. 如果需要简单数值计算，调用 calculator。
8. 如果系统提示中的“按需加载”技能摘要与当前任务相关，先用 read_file 读取对应路径，再应用技能内容。
9. 工具返回后，要综合结果给出清晰、简洁、专业的最终回答。
   - 结论必须前置，先写结论，再写支撑数据。
   - 每个投资判断都必须引用至少 2 个具体数据点，例如最新收盘价、近20日/60日涨跌幅、均线、总市值或行业。
   - 对金融投资类问题，优先走 hold_analysis_pipeline 生成最终回答，不要只拼接工具原文。
10. 如果信息不足，明确说明不确定性，不要编造事实。
11. 技术面与基本面必须交叉验证，不能只靠 K 线与均线下结论。
    - 单股趋势/持有类问题：拿到价格数据后，务必再用 stock_a_valuation（TTM PE/PB/PS 及历史分位）、stock_a_financial_indicators（盈利/毛利/营收增速）交叉验证；估值底（低 PE 历史分位）是技术低点之外的重要下方支撑，必须纳入分析。
    - 批价（如飞天批价）、渠道库存、提价/出厂价上调预期、业绩（如 Q3 报表）等若无直接工具数据，须列为关键定性观察点并标注"数据缺失/来源未知"，不得编造。
    - 乐观情景的触发条件须包含"提价/出厂价上调预期"；悲观情景须包含"业绩不及预期（动销走弱、报表继续承压）"风险。
    - 时间轴以工具返回的 latest_trade_date / as_of 为"当前"，推算"未来两个月"≈其后约 60 个交易日（约 2 个月），不要臆测具体月份；区间与情景的时间标签须与数据日期一致，避免月份错位。

统一引用规范：
- 行情 / 资金：引用东方财富、通达信或 Wind，必须标注日期。
- 公司公告 / 互动：引用巨潮资讯网或深交所互动易，必须标注公告日期或互动日期。
- 行业数据：引用 Omdia 或中国通信院，必须标注报告季度或发布日期。
- 若工具数据没有对应来源或日期，必须写“来源/日期缺失”，不得伪造来源。
""".strip()

def convert_messages(messages: list[BaseMessage], memories=None, compression_mode="auto") -> list[dict]:
    builder = ContextBuilder(SYSTEM_PROMPT)
    return builder.build_messages(messages, memories=memories, compression_mode=compression_mode)

def _internal_usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# 可重试的瞬时错误：限流 / 服务端 5xx / 连接或超时。
# 注意：413（prompt_too_long）不在其中，以便 litellm_glm_call 继续走压缩回退。
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "GatewayTimeoutError",
        "Timeout",
        "APIStatusError",
    }
)


def _internal_is_transient_error(exc: Exception) -> bool:
    """判断是否为可重试的瞬时错误（限流 / 5xx / 连接或超时）。"""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    name = type(exc).__name__
    if name in _TRANSIENT_EXCEPTION_NAMES or "Timeout" in name or "Connection" in name:
        return True
    # litellm 可能把底层网络错误包装成 APIError；按模块/名称兜底
    module = getattr(type(exc), "__module__", "") or ""
    if "litellm" in module or "openai" in module:
        lowered = name.lower()
        if any(
            token in lowered
            for token in (
                "timeout",
                "connection",
                "ratelimit",
                "internalserver",
                "serviceunavailable",
                "gatewaytimeout",
            )
        ):
            return True
    return False


def _internal_max_retries() -> int:
    # 默认 2 次：单次调用已带 FINABOT_LLM_TIMEOUT_SECONDS 超时，过多重试会把
    # 响应预算（FINABOT_RESPONSE_TIMEOUT_SECONDS）吃光，反而触发整体超时。
    try:
        value = int(os.getenv("FINABOT_LLM_MAX_RETRIES", "2"))
    except ValueError:
        value = 2
    return max(1, min(value, 4))


def _internal_retry_base_seconds() -> float:
    try:
        value = float(os.getenv("FINABOT_LLM_RETRY_BASE_SECONDS", "2"))
    except ValueError:
        value = 2.0
    return max(0.5, min(value, 30.0))


async def _internal_acompletion(
    settings: LLMSettings,
    messages: list[dict],
    tools,
    *,
    retry: bool,
):
    max_attempts = _internal_max_retries()
    base_delay = _internal_retry_base_seconds()
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        started_at = utc_timestamp()
        started = perf_counter()
        request_kwargs: dict[str, Any] = {
            "model": settings.litellm_model,
            "api_key": settings.api_key,
            "messages": messages,
            "tools": tools,
            "temperature": 0.1,
            "timeout": settings.timeout_seconds,
        }
        if settings.api_base:
            # 自定义 OpenAI 兼容端点（如 https://inferaichat.com/v1）
            request_kwargs["api_base"] = settings.api_base
        try:
            response = await asyncio.wait_for(
                acompletion(**request_kwargs),
                timeout=settings.timeout_seconds,
            )
        except Exception as exc:
            last_exc = exc
            LLM_METRICS.record(
                LLMCallMetric(
                    model=settings.litellm_model,
                    started_at=started_at,
                    latency_ms=round((perf_counter() - started) * 1000, 2),
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    success=False,
                    retry=attempt > 1,
                    error_type=type(exc).__name__,
                )
            )
            # 不可重试（客户端错误 / prompt_too_long）或已耗尽重试次数，直接抛出，
            # 交由 litellm_glm_call 的提示压缩回退等上层逻辑处理。
            if attempt >= max_attempts or not _internal_is_transient_error(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), 30.0) + random.uniform(0, 1.0)
            await asyncio.sleep(delay)
            continue

        usage = getattr(response, "usage", None)
        prompt_tokens = _internal_usage_value(usage, "prompt_tokens")
        completion_tokens = _internal_usage_value(usage, "completion_tokens")
        total_tokens = _internal_usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
        LLM_METRICS.record(
            LLMCallMetric(
                model=settings.litellm_model,
                started_at=started_at,
                latency_ms=round((perf_counter() - started) * 1000, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                success=True,
                retry=attempt > 1,
            )
        )
        return response

    # 兜底：循环内失败必已 raise；此处仅在极端情况下触发。
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unexpected empty completion result")


async def _internal_acompletion_stream(
    settings: LLMSettings,
    messages: list[dict],
    tools,
    sink,
    stream_label: str,
):
    """以 stream=True 调用 LLM，把每个文本 delta 实时转发给 sink。

    同时累积 content 与 tool_calls（GLM 的工具调用参数会分片到达），返回一个
    与 `response.choices[0].message` 同形状的 SimpleNamespace，保证
    `call_llm_node` 的后续归一化逻辑无需区分流式/非流式。
    """
    started_at = utc_timestamp()
    started = perf_counter()
    request_kwargs: dict[str, Any] = {
        "model": settings.litellm_model,
        "api_key": settings.api_key,
        "messages": messages,
        "tools": tools,
        "temperature": 0.1,
        "timeout": settings.timeout_seconds,
        "stream": True,
    }
    if settings.api_base:
        request_kwargs["api_base"] = settings.api_base

    content_parts: list[str] = []
    tool_call_fragments: dict[int, dict[str, str]] = {}
    order: list[int] = []

    try:
        # stream=True 时 acompletion 返回异步生成器（AsyncStream），不能直接
        # await；这里对每个分片单独套超时，避免整体流被某个慢分片拖死。
        chunks = acompletion(**request_kwargs)
    except Exception as exc:
        LLM_METRICS.record(
            LLMCallMetric(
                model=settings.litellm_model,
                started_at=started_at,
                latency_ms=round((perf_counter() - started) * 1000, 2),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                retry=False,
                error_type=type(exc).__name__,
            )
        )
        raise

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(
                    chunks.__anext__(),
                    timeout=settings.timeout_seconds,
                )
            except StopAsyncIteration:
                break
            delta = chunk.choices[0].delta if (chunk.choices and chunk.choices[0]) else None
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
                try:
                    await sink(stream_label, content)
                except Exception:
                    # 进度推送失败不应中断 LLM 输出本身
                    pass
            raw_tool_calls = getattr(delta, "tool_calls", None)
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    idx = tc.index if isinstance(tc.index, int) else 0
                    if idx not in tool_call_fragments:
                        tool_call_fragments[idx] = {"name": "", "arguments": ""}
                        order.append(idx)
                    fn = getattr(tc, "function", None) or {}
                    if getattr(fn, "name", None):
                        tool_call_fragments[idx]["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        tool_call_fragments[idx]["arguments"] += fn.arguments
    except Exception as exc:
        LLM_METRICS.record(
            LLMCallMetric(
                model=settings.litellm_model,
                started_at=started_at,
                latency_ms=round((perf_counter() - started) * 1000, 2),
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                retry=False,
                error_type=type(exc).__name__,
            )
        )
        raise

    tool_calls = [
        {
            "id": f"finabot_stream_{idx}",
            "function": {"name": tool_call_fragments[idx]["name"], "arguments": tool_call_fragments[idx]["arguments"]},
            "type": "function",
        }
        for idx in order
    ]

    usage = getattr(chunks, "usage", None)
    prompt_tokens = _internal_usage_value(usage, "prompt_tokens")
    completion_tokens = _internal_usage_value(usage, "completion_tokens")
    total_tokens = _internal_usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
    LLM_METRICS.record(
        LLMCallMetric(
            model=settings.litellm_model,
            started_at=started_at,
            latency_ms=round((perf_counter() - started) * 1000, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            success=True,
            retry=False,
        )
    )
    return SimpleNamespace(content="".join(content_parts), tool_calls=tool_calls)


async def litellm_glm_call(
    messages: list[BaseMessage],
    tools=None,
    memories=None,
    stream_label: str | None = None,
    system_prompt: str | None = None,
):
    settings = get_llm_settings()
    if system_prompt is not None:
        # 覆盖默认 supervisor 系统提示（如会话摘要器等独立角色），
        # 同时仍走同一套消息转换与压缩管线。
        messages_dict = ContextBuilder(system_prompt.strip()).build_messages(
            messages, memories=memories
        )
    else:
        messages_dict = convert_messages(messages, memories=memories)
    has_tools = "tools" if tools else "no-tools"
    _debug_timing(f"llm:start model={settings.litellm_model} {has_tools} msgs={len(messages_dict)}")
    started = perf_counter()

    sink = get_token_sink()
    if sink is not None and is_streamable_label(stream_label):
        try:
            return await _internal_acompletion_stream(
                settings, messages_dict, tools, sink, stream_label
            )
        except Exception as exc:
            # 流式失败（如连接中断）时回退到非流式完整调用，保证结果不丢
            if not _internal_is_prompt_too_long(exc):
                _debug_timing(f"llm:stream-error model={settings.litellm_model} {type(exc).__name__}")
                response = await _internal_acompletion(settings, messages_dict, tools, retry=False)
                return response.choices[0].message
            # prompt_too_long 交给下方压缩重试路径处理
    try:
        response = await _internal_acompletion(settings, messages_dict, tools, retry=False)
    except Exception as exc:
        if not _internal_is_prompt_too_long(exc):
            _debug_timing(f"llm:error model={settings.litellm_model} after={round((perf_counter() - started) * 1000)}ms {type(exc).__name__}")
            raise
        retry_builder = (
            ContextBuilder(system_prompt.strip())
            if system_prompt is not None
            else None
        )
        if retry_builder is not None:
            retry_messages = retry_builder.build_messages(
                messages, memories=memories, compression_mode="reactive"
            )
        else:
            retry_messages = convert_messages(
                messages, memories=memories, compression_mode="reactive"
            )
        response = await _internal_acompletion(
            settings,
            retry_messages,
            tools,
            retry=True,
        )
    _debug_timing(f"llm:done model={settings.litellm_model} elapsed={round((perf_counter() - started) * 1000)}ms")

    return response.choices[0].message


def _internal_is_prompt_too_long(exc: Exception) -> bool:
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return status_code == 413 or "prompt_too_long" in text or "context length" in text
