import asyncio
import json
import os
import re
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import tool_call

from finabot.agents.llm import SINGLE_AGENT_SYSTEM_PROMPT, litellm_glm_call, _debug_timing
from finabot.agents.state import AgentState
from finabot.agents.analysts.market_analyst import _internal_call_market_analyst
from finabot.agents.analysts.news_analyst import _internal_call_news_analyst
from finabot.agents.hold_pipeline import run_hold_analysis_pipeline
from finabot.agents.researchers.researchers import _internal_call_researchers
from finabot.agents.analysts.fundamental_analyst import _internal_call_fundamental_analyst
from finabot.agents.evidence import register_subagent_evidence, register_tool_evidence
from finabot.agents.refusal import classify_question
from finabot.agents.schema import structured_state_update
from finabot.tools.base import get_tools


tools = get_tools()


_TOOL_CALL_TEXT_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][\w\-]*)\s*(?:```(?:html)?\s*)?(?:<tool_call>)?.*?"
    r"<arg_key>\s*(?P<key>[^<]+?)\s*</arg_key>\s*"
    r"<arg_value>\s*(?P<value>.*?)\s*</arg_value>.*?(?:</tool_call>)?",
    re.IGNORECASE | re.DOTALL,
)

# 多工具/多参数文本回退解析：GLM 偶尔把工具调用当纯文本回显，可能是
# 单个或多个 `<tool_call>` 块，每块含多对 `<arg_key>/<arg_value>`。
# 工具名可在块前（`name <tool_call>...`）或块内 `<arg_key>` 之前。
_TOOL_CALL_BLOCK_WITH_NAME_PATTERN = re.compile(
    r"(?:(?P<name>[A-Za-z_][\w\-]*)\s*)?<tool_call>(?P<body>.*?)</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_ARG_PAIR_PATTERN = re.compile(
    r"<arg_key>\s*(?P<key>[^<]+?)\s*</arg_key>\s*<arg_value>\s*(?P<value>.*?)\s*</arg_value>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_NAME_PATTERN = re.compile(r"<function>\s*(?P<name>[^<]+?)\s*</function>", re.IGNORECASE)


def _internal_normalize_tool_name(name) -> str:
    if name is None:
        return ""
    return str(name).strip()


def _internal_latest_user_message(state: AgentState) -> str:
    """与子代理节点路由保持一致：子代理消费最新人类消息，而非工具调用参数。"""
    messages = state.get("messages", []) or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            content = getattr(message, "content", "") or ""
            if content:
                return str(content)
    if messages:
        return str(getattr(messages[-1], "content", "") or "")
    return ""


_SUB_AGENT_NAMES = {
    "fundamental_analyst",
    "market_analyst",
    "news_analyst",
    "researchers",
    "hold_analysis_pipeline",
}


async def _call_with_timeout(coro, name: str, timeout: float | None = None) -> str:
    """Wrap a sub-agent call with a configurable timeout.

    On timeout the function returns a structured placeholder so the
    supervisor / summary manager can degrade confidence instead of
    crashing the whole graph.
    """
    if timeout is None:
        timeout = float(os.getenv("FINABOT_SUBAGENT_TIMEOUT_SECONDS", "60"))
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        _debug_timing(f"subagent:timeout name={name} timeout={timeout}s")
        return (
            f"[subagent_timeout:{name}] 该子代理未能在 {timeout:.0f}s 内完成，"
            f"结论置信度降级。"
        )


async def _internal_invoke_sub_agent(name: str, state: AgentState, call: dict | None = None) -> tuple[str, dict]:
    """以与 graph 节点路由完全相同的上下文调用子代理。

    子代理同时注册为节点和工具；无论 supervisor 的 tool_call 走专用节点
    还是落入通用工具节点（多工具并发等场景），都必须共享 state 报告、
    AKShare 缓存。返回 (结果文本, 需写回的 state 增量)。

    每个子代理调用包超时保护（FINABOT_SUBAGENT_TIMEOUT_SECONDS，默认 60s），
    超时返回结构化占位并降级置信度，避免单节点失败导致整轮崩溃。

    多空辩论与总结已折叠进 hold_analysis_pipeline，不再作为独立的
    supervisor 路由子代理，因此这里也不再单独处理 bull/bear/summary。
    """
    expression = _internal_latest_user_message(state)
    args = (call or {}).get("args", {}) or {}

    if name == "fundamental_analyst":
        raw = await _call_with_timeout(
            _internal_call_fundamental_analyst(expression, state.setdefault("akshare_cache", {})),
            name,
        )
        display, update = structured_state_update("fundamental_analyst", str(raw), state, state.get("as_of"))
        update["fundamentals_report"] = display
        return display, update

    if name == "market_analyst":
        raw = await _call_with_timeout(
            _internal_call_market_analyst(expression),
            name,
        )
        display, update = structured_state_update("market_analyst", str(raw), state, state.get("as_of"))
        update["market_report"] = display
        return display, update

    if name == "news_analyst":
        raw = await _call_with_timeout(
            _internal_call_news_analyst(expression, state.setdefault("akshare_cache", {})),
            name,
        )
        display, update = structured_state_update("news_analyst", str(raw), state, state.get("as_of"))
        update["news_report"] = display
        return display, update

    if name == "researchers":
        raw = await _call_with_timeout(
            _internal_call_researchers(expression),
            name,
        )
        display, update = structured_state_update("researchers", str(raw), state, state.get("as_of"))
        return display, update

    if name == "hold_analysis_pipeline":
        debate_mode = bool(args.get("debate_mode", False))
        pipeline_result = await _call_with_timeout(
            run_hold_analysis_pipeline(
                expression,
                {
                    "market_report": state.get("market_report", ""),
                    "memories": state.get("memories", []),
                    "akshare_cache": state.setdefault("akshare_cache", {}),
                    "as_of": state.get("as_of"),
                },
                debate_mode=debate_mode,
            ),
            name,
        )
        if isinstance(pipeline_result, str):
            # 超时占位：退化为降级结果，避免下游对字符串调用 .get()
            placeholder = pipeline_result
            result = {
                "fundamentals_report": placeholder,
                "news_report": placeholder,
                "bull_report": placeholder,
                "bear_report": placeholder,
                "summary_report": placeholder,
                "claims": [],
                "risk_flags": [],
            }
        else:
            result = pipeline_result
        content = result.get("debate_report") if debate_mode else result["summary_report"]
        update: dict = {
            "fundamentals_report": result["fundamentals_report"],
            "news_report": result["news_report"],
            "bull_report": result["bull_report"],
            "bear_report": result["bear_report"],
        }
        claims = result.get("claims") or []
        if claims:
            update["claims"] = list(state.get("claims", []) or []) + list(claims)
        risk_flags = result.get("risk_flags") or []
        if risk_flags:
            update["risk_flags"] = list(state.get("risk_flags", []) or []) + list(risk_flags)
        return str(content), update

    raise ValueError(f"unknown sub-agent: {name}")


def normalize_tool_call(call):
    """Convert LiteLLM/OpenAI tool-call objects to the LangChain ToolCall shape."""
    if isinstance(call, dict):
        if "function" in call:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments", {})
            tool_name = _internal_normalize_tool_name(function.get("name", ""))
            if not tool_name:
                return None
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError:
                    arguments = {"expression": raw_arguments}
            else:
                arguments = raw_arguments or {}
            return tool_call(
                name=tool_name,
                args=arguments,
                id=call.get("id"),
            )

        raw_arguments = call.get("args", {})
        tool_name = _internal_normalize_tool_name(call.get("name", ""))
        if not tool_name:
            return None
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                arguments = {"expression": raw_arguments}
        else:
            arguments = raw_arguments or {}

        return tool_call(
            name=tool_name,
            args=arguments,
            id=call.get("id"),
        )

    function = getattr(call, "function", None)
    raw_arguments = getattr(function, "arguments", {})
    tool_name = _internal_normalize_tool_name(getattr(function, "name", ""))
    if not tool_name:
        return None
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            arguments = {"expression": raw_arguments}
    else:
        arguments = raw_arguments or {}

    return tool_call(
        name=tool_name,
        args=arguments,
        id=getattr(call, "id", None),
    )


def _internal_parse_tool_call_body(name: str, body: str):
    """解析工具调用体（含参数对），返回 tool_call 或 None。

    ``name`` 为已从块前捕获的工具名；为空时回退：`<function>` 包裹 →
    body 内首个 `<arg_key>` 前的最后一个函数式标识符。
    """
    arguments: dict[str, str] = {}
    for pair in _ARG_PAIR_PATTERN.finditer(body):
        key = pair.group("key").strip()
        value = pair.group("value").strip()
        if key:
            arguments[key] = value
    if not arguments:
        return None

    if not name:
        fn_match = _FUNCTION_NAME_PATTERN.search(body)
        if fn_match:
            name = fn_match.group("name").strip()
    if not name:
        before_args = body.split("<arg_key", 1)[0]
        tokens = re.findall(r"[A-Za-z_][\w\-]*", before_args)
        name = tokens[-1].strip() if tokens else ""
    if not name:
        return None
    return tool_call(name=name, args=arguments, id=None)


def extract_tool_calls_from_content(content: str):
    """Fallback parser for assistant text that contains serialized tool markup.

    支持多个 `<tool_call>` 块（工具名可在块前或块内 `<arg_key>` 之前）、每块多对
    `<arg_key>/<arg_value>`；无 `<tool_call>` 包裹时回退为对整个内容解析单个
    工具调用（兼容旧单工具行为）。
    """
    if not content:
        return []

    calls = []
    for match in _TOOL_CALL_BLOCK_WITH_NAME_PATTERN.finditer(content):
        call = _internal_parse_tool_call_body(
            (match.group("name") or "").strip(),
            match.group("body"),
        )
        if call is not None:
            calls.append(call)
    if calls:
        return calls

    call = _internal_parse_tool_call_body("", content)
    return [call] if call is not None else []


def _strip_tool_call_markup(text: str) -> str:
    """清掉正文中残留的 <tool_call> / <function> 工具调用标记（模型偶尔把工具调用当纯文本回显）。"""
    if not text:
        return ""
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<function>.*?</function>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # 去掉可能遗留的孤立 <arg_key>/<arg_value> 标记
    cleaned = re.sub(r"</?arg_(?:key|value)>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def format_tools(single_agent: bool = False):
    formatted_tools = []

    tool_list = tools
    if single_agent:
        # 单 Agent 对照组：supervisor 不再看到任何子代理工具，只保留数据/计算工具
        tool_list = [t for t in tools if t.name not in _SUB_AGENT_NAMES]

    for t in tool_list:
        args_schema = getattr(t, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            parameters = args_schema.model_json_schema()
        elif args_schema is not None and hasattr(args_schema, "schema"):
            parameters = args_schema.schema()
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
            }

        formatted_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": parameters,
                },
            }
        )

    return formatted_tools


def _internal_supervisor_rounds(state: AgentState) -> int:
    """已发生的 supervisor 工具调用轮次（带 tool_calls 的 AIMessage 数量）。"""
    return sum(
        1
        for message in state.get("messages", [])
        if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None)
    )


def _internal_refusal_note_for(state: AgentState) -> str:
    """用户问题命中合规边界时返回合规说明，否则返回空字符串。"""
    question = _internal_latest_user_message(state)
    if not question:
        return ""
    decision = classify_question(question)
    if decision.level == "safe":
        return ""
    return (
        "【合规边界】用户请求命中「具体买卖/仓位/收益承诺」边界"
        f"（判定：{decision.reason}）。你必须："
        "1) 明确说明无法提供个性化买卖/仓位建议；"
        "2) 转为一堂风险教育与研究方法课（如何看估值、如何计算收益率、如何理解风险）；"
        "3) 不给出任何具体股票、具体价位、具体仓位比例或收益预期。"
    )


async def call_llm_node(state: AgentState, single_agent: bool = False):
    # 轮次预算：第三方端点偏慢且模型可能反复派发子代理，若不限轮次会在
    # FINABOT_RESPONSE_TIMEOUT_SECONDS 内跑不完。超预算后强制让 supervisor 直接
    # 给出最终回答（去掉 tools 并丢弃残留工具调用），把总 LLM 调用数封顶。
    max_rounds = max(1, int(os.getenv("FINABOT_MAX_LLM_ROUNDS", "6")))
    rounds = _internal_supervisor_rounds(state)
    force_final = rounds >= max_rounds
    if force_final:
        _debug_timing(f"llm:force_final rounds={rounds} max={max_rounds}")

    messages = state["messages"]
    # 合规拒绝路径：用户问题命中"具体买卖/仓位/收益承诺"边界时，
    # 在消息序列最前面注入合规说明，让 supervisor 把回答转为一般教育。
    refusal_note = _internal_refusal_note_for(state)
    if refusal_note:
        messages = [SystemMessage(content=refusal_note)] + list(messages)
    if force_final:
        # 轮次耗尽时追加强制合成指令，避免模型在巨大上下文中迷路吐出问候
        messages = list(messages) + [
            HumanMessage(content=(
                "[系统强制指令] 工具调用轮次已达上限，你必须立刻综合前面已获取的全部数据，"
                "直接给出最终分析回答。不要再发起任何工具调用。\n"
                "请根据市场行情、估值（TTM PE/PB 及历史分位）、财务指标、新闻分析等已有信息，"
                "按结论前置的专业格式给出完整回答。如果某类数据缺失，注明'数据缺失'即可。"
            ))
        ]

    system_prompt = SINGLE_AGENT_SYSTEM_PROMPT if single_agent else None
    call_kwargs: dict = {
        "messages": messages,
        "tools": None if force_final else format_tools(single_agent=single_agent),
        "memories": state.get("memories"),
        "stream_label": "supervisor",
    }
    if system_prompt is not None:
        call_kwargs["system_prompt"] = system_prompt
    msg = await litellm_glm_call(**call_kwargs)

    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    tool_calls = [call for call in (normalize_tool_call(call) for call in raw_tool_calls) if call is not None]
    content = getattr(msg, "content", "") or ""

    if force_final:
        # 预算耗尽：不再执行任何工具调用（含 GLM 以纯文本回退的 <tool_call> 标记），
        # 直接综合已获取信息作答，避免无限循环导致响应超时。同时清掉正文中残留的
        # 工具调用标记，避免把 <tool_call> 原样回显给用户。
        tool_calls = []
        content = _strip_tool_call_markup(content)
        if not content.strip():
            content = "（已达最大分析轮次，以下为基于已获取信息的综合判断）"
    elif not tool_calls:
        tool_calls = extract_tool_calls_from_content(content)
        if tool_calls:
            content = ""

    # 部分模型/兜底解析会产出缺 id 的 tool_call；补齐后 assistant.tool_calls
    # 与后续 ToolMessage 才能稳定配对（OpenAI 兼容 API 要求两侧 id 一致）。
    for position, call in enumerate(tool_calls):
        if not call.get("id"):
            tool_calls[position] = {**call, "id": f"finabot_call_{position}"}

    ai_msg = AIMessage(
        content=content,
        tool_calls=tool_calls
    )
    return {"messages": [ai_msg]}


async def call_tool_node(state: AgentState):
    last = state["messages"][-1]
    # 预置缓存，避免多个子代理在 gather 中并发 setdefault 时各自新建 dict 互相覆盖
    state.setdefault("akshare_cache", {})
    registry = state.setdefault("evidence_registry", {})

    calls = list(enumerate(last.tool_calls))

    async def _run_one(index, call):
        normalized_call = normalize_tool_call(call)
        raw_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        call_id = raw_id or f"finabot_call_{index}"

        # 每个 tool_call 必须有对应的 ToolMessage，否则下一条请求会因
        # "assistant.tool_calls 缺少 tool 结果"被 OpenAI 兼容 API 拒绝。
        if normalized_call is None:
            return index, ToolMessage(content="工具调用格式无法解析。", tool_call_id=call_id), None

        tool_name = normalized_call["name"]

        # 子代理双注册统一：即便与多个工具并发执行，也按节点路由相同的
        # 上下文执行并写回报告/缓存增量（与节点路由行为一致）。
        if tool_name in _SUB_AGENT_NAMES:
            _debug_timing(f"tool:start sub_agent={tool_name}")
            started = perf_counter()
            try:
                result_text, updates = await _internal_invoke_sub_agent(tool_name, state, normalized_call)
            except Exception as exc:
                _debug_timing(f"tool:error sub_agent={tool_name} after={round((perf_counter() - started) * 1000)}ms {type(exc).__name__}")
                return (
                    index,
                    ToolMessage(content=f"子代理 {tool_name} 执行失败：{exc}", tool_call_id=call_id),
                    None,
                )
            _debug_timing(f"tool:done sub_agent={tool_name} elapsed={round((perf_counter() - started) * 1000)}ms")
            register_subagent_evidence(registry, tool_name, str(result_text), state.get("as_of"))
            updates = dict(updates or {})
            updates["evidence_registry"] = registry
            return index, ToolMessage(content=str(result_text), tool_call_id=call_id), updates

        t = next((x for x in tools if x.name == tool_name), None)
        if t is None:
            return index, ToolMessage(content=f"未知工具：{tool_name}。", tool_call_id=call_id), None
        _debug_timing(f"tool:start {tool_name}")
        started = perf_counter()
        try:
            res = await t.ainvoke(normalized_call["args"])
        except Exception as exc:
            # 单工具失败隔离：不再像串行版本那样让整轮崩溃
            _debug_timing(f"tool:error {tool_name} after={round((perf_counter() - started) * 1000)}ms {type(exc).__name__}")
            return (
                index,
                ToolMessage(content=f"工具 {tool_name} 执行失败：{exc}", tool_call_id=call_id),
                None,
            )
        _debug_timing(f"tool:done {tool_name} elapsed={round((perf_counter() - started) * 1000)}ms")
        register_tool_evidence(registry, tool_name, str(res))
        return (
            index,
            ToolMessage(content=str(res), tool_call_id=call_id),
            {"evidence_registry": registry},
        )

    # 并发执行所有 tool_call，把多工具批次的累计耗时压到单次最大值；
    # 结果按原 call 顺序组装，保证确定性。state 增量按 key 合并（last-writer-wins）。
    results = await asyncio.gather(*(_run_one(index, call) for index, call in calls))
    ordered = sorted(results, key=lambda item: item[0])
    tool_results = [message for _, message, _ in ordered]
    state_updates: dict = {}
    for _, _message, updates in ordered:
        if updates:
            state_updates.update(updates)
    return {"messages": tool_results, **state_updates}

