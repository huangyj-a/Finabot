from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from finabot.agents.nodes import (
    _call_with_timeout,
    _internal_latest_user_message,
    _pipeline_timeout,
    call_llm_node as call_supervisor_node,
    call_tool_node,
)
from finabot.agents.state import AgentState
from finabot.agents.analysts import (
    _internal_call_fundamental_analyst,
    _internal_call_news_analyst,
    market_analyst,
)
from finabot.agents.hold_pipeline import run_hold_analysis_pipeline
from finabot.agents.failure import injected_failure
from finabot.agents.researchers import researchers
from finabot.agents.schema import structured_state_update
from finabot.graph.router import classify_intent


def _internal_failure_update(node_name: str, state: AgentState) -> dict | None:
    """返回失败注入的 state 增量；未命中注入时返回 None。"""
    fail = injected_failure(node_name)
    if fail is None:
        return None
    return {
        "messages": [AIMessage(content=fail)],
        "risk_flags": list(state.get("risk_flags", []) or []) + [fail],
    }


def _internal_router_node(state: AgentState) -> dict:
    """规则预路由节点：总是写入 debate_mode（LangGraph 节点不可返回空更新），
    命中持有分析且含辩论关键词时为 True。"""
    question = _internal_latest_user_message(state)
    _target, debate = classify_intent(question)
    return {"debate_mode": debate}


def _internal_route_intent(state: AgentState) -> str:
    """条件边：规则命中直接进对应节点，否则回落到 LLM supervisor。"""
    question = _internal_latest_user_message(state)
    target, _ = classify_intent(question)
    return target or "supervisor"


async def _internal_market_analyst_node(state: AgentState):
    fail = _internal_failure_update("market_analyst", state)
    if fail is not None:
        fail["market_report"] = fail["messages"][0].content
        return fail
    expression = _internal_latest_user_message(state)
    raw = await _call_with_timeout(
        market_analyst.ainvoke({"expression": expression}),
        "market_analyst",
    )
    display, update = structured_state_update("market_analyst", str(raw), state, state.get("as_of"))
    update.update({"messages": [AIMessage(content=display)], "market_report": display})
    return update


async def _internal_fundamental_analyst_node(state: AgentState):
    fail = _internal_failure_update("fundamental_analyst", state)
    if fail is not None:
        fail["fundamentals_report"] = fail["messages"][0].content
        return fail
    expression = _internal_latest_user_message(state)
    raw = await _call_with_timeout(
        _internal_call_fundamental_analyst(expression, state.setdefault("akshare_cache", {})),
        "fundamental_analyst",
    )
    display, update = structured_state_update("fundamental_analyst", str(raw), state, state.get("as_of"))
    update.update({"messages": [AIMessage(content=display)], "fundamentals_report": display})
    return update


async def _internal_news_analyst_node(state: AgentState):
    fail = _internal_failure_update("news_analyst", state)
    if fail is not None:
        fail["news_report"] = fail["messages"][0].content
        return fail
    expression = _internal_latest_user_message(state)
    raw = await _call_with_timeout(
        _internal_call_news_analyst(expression, state.setdefault("akshare_cache", {})),
        "news_analyst",
    )
    display, update = structured_state_update("news_analyst", str(raw), state, state.get("as_of"))
    update.update({"messages": [AIMessage(content=display)], "news_report": display})
    return update


async def _internal_researchers_node(state: AgentState):
    fail = _internal_failure_update("researchers", state)
    if fail is not None:
        return fail
    expression = _internal_latest_user_message(state)
    raw = await _call_with_timeout(
        researchers.ainvoke({"expression": expression}),
        "researchers",
    )
    display, update = structured_state_update("researchers", str(raw), state, state.get("as_of"))
    update.update({"messages": [AIMessage(content=display)]})
    return update


def _internal_extract_debate_mode(state: AgentState) -> bool:
    """读取 debate_mode：优先 supervisor tool_call 参数（LLM 路由路径），
    其次规则路由节点写入的状态（router 短路路径）。"""
    last = state["messages"][-1]
    for tool_call in getattr(last, "tool_calls", []) or []:
        name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
        if name == "hold_analysis_pipeline":
            args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            return bool((args or {}).get("debate_mode", False))
    return bool(state.get("debate_mode", False))


async def _internal_hold_analysis_pipeline_node(state: AgentState):
    fail = _internal_failure_update("hold_analysis_pipeline", state)
    if fail is not None:
        fail["summary_report"] = fail["messages"][0].content
        return fail
    expression = _internal_latest_user_message(state)
    debate_mode = _internal_extract_debate_mode(state)
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
        "hold_analysis_pipeline",
        timeout=_pipeline_timeout(),
    )
    if isinstance(pipeline_result, str):
        placeholder = pipeline_result
        result = {
            "fundamentals_report": placeholder,
            "news_report": placeholder,
            "bull_report": placeholder,
            "bear_report": placeholder,
            "summary_report": placeholder,
            "debate_report": placeholder,
            "claims": [],
            "risk_flags": [],
        }
    else:
        result = pipeline_result
    content = result.get("debate_report") if debate_mode else result["summary_report"]
    if not content:
        content = result.get("summary_report") or ""  # debate_report 缺失时回退 summary
    update: dict = {
        "messages": [AIMessage(content=str(content))],
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
    return update


def _internal_make_route_supervisor(single_agent: bool):
    """路由闭包：单 Agent 模式不返回任何子代理分支，只有 tool/end。"""

    def _route(state: AgentState):
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "end"

        if not single_agent and len(tool_calls) == 1:
            tool_name = str(tool_calls[0].get("name") if isinstance(tool_calls[0], dict) else getattr(tool_calls[0], "name", "") or "")
            if tool_name == "market_analyst":
                return "market_analyst"
            if tool_name == "fundamental_analyst":
                return "fundamental_analyst"
            if tool_name == "news_analyst":
                return "news_analyst"
            if tool_name == "researchers":
                return "researchers"
            if tool_name == "hold_analysis_pipeline":
                return "hold_analysis_pipeline"

        return "tool"

    return _route


def build_graph(checkpointer=None, single_agent: bool = False):
    from functools import partial

    g = StateGraph(AgentState)
    g.add_node("supervisor", partial(call_supervisor_node, single_agent=single_agent))
    if not single_agent:
        g.add_node("market_analyst", _internal_market_analyst_node)
        g.add_node("fundamental_analyst", _internal_fundamental_analyst_node)
        g.add_node("news_analyst", _internal_news_analyst_node)
        g.add_node("researchers", _internal_researchers_node)
        g.add_node("hold_analysis_pipeline", _internal_hold_analysis_pipeline_node)
    g.add_node("tool", call_tool_node)

    route_map = {
        "tool": "tool",
        "end": END,
    }
    if not single_agent:
        route_map.update({
            "market_analyst": "market_analyst",
            "fundamental_analyst": "fundamental_analyst",
            "news_analyst": "news_analyst",
            "researchers": "researchers",
            "hold_analysis_pipeline": "hold_analysis_pipeline",
        })

    g.add_conditional_edges(
        "supervisor",
        _internal_make_route_supervisor(single_agent),
        route_map,
    )

    if not single_agent:
        # 规则预路由：高置信意图（持有分析 / 市场分析）直接短路进对应节点，
        # 省一次 supervisor LLM 往返；其余回落 LLM supervisor。
        g.add_node("router", _internal_router_node)
        g.add_edge(START, "router")
        g.add_conditional_edges(
            "router",
            _internal_route_intent,
            {
                "hold_analysis_pipeline": "hold_analysis_pipeline",
                "market_analyst": "market_analyst",
                "supervisor": "supervisor",
            },
        )
    else:
        # 单 Agent 对照组：不挂规则路由，保持 START → supervisor 原拓扑
        g.add_edge(START, "supervisor")

    if not single_agent:
        g.add_edge("market_analyst", "supervisor")
        g.add_edge("fundamental_analyst", "supervisor")
        g.add_edge("news_analyst", "supervisor")
        g.add_edge("researchers", "supervisor")
        g.add_edge("hold_analysis_pipeline", "supervisor")
    g.add_edge("tool", "supervisor")
    return g.compile(checkpointer=checkpointer)
