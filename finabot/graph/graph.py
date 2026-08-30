from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from finabot.agents.nodes import (
    _call_with_timeout,
    _internal_latest_user_message,
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
from finabot.agents.researchers import researchers


async def _internal_market_analyst_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await _call_with_timeout(
        market_analyst.ainvoke({"expression": expression}),
        "market_analyst",
    )
    return {"messages": [AIMessage(content=str(result))], "market_report": str(result)}


async def _internal_fundamental_analyst_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await _call_with_timeout(
        _internal_call_fundamental_analyst(expression, state.setdefault("akshare_cache", {})),
        "fundamental_analyst",
    )
    return {"messages": [AIMessage(content=str(result))], "fundamentals_report": str(result)}


async def _internal_news_analyst_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await _call_with_timeout(
        _internal_call_news_analyst(expression, state.setdefault("akshare_cache", {})),
        "news_analyst",
    )
    return {"messages": [AIMessage(content=str(result))], "news_report": str(result)}


async def _internal_researchers_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await _call_with_timeout(
        researchers.ainvoke({"expression": expression}),
        "researchers",
    )
    return {"messages": [AIMessage(content=str(result))]}


def _internal_extract_debate_mode(state: AgentState) -> bool:
    """从 supervisor 最近一次对 hold_analysis_pipeline 的 tool_call 中读取 debate_mode。"""
    last = state["messages"][-1]
    for tool_call in getattr(last, "tool_calls", []) or []:
        name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
        if name == "hold_analysis_pipeline":
            args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            return bool((args or {}).get("debate_mode", False))
    return False


async def _internal_hold_analysis_pipeline_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    debate_mode = _internal_extract_debate_mode(state)
    pipeline_result = await _call_with_timeout(
        run_hold_analysis_pipeline(
            expression,
            {
                "market_report": state.get("market_report", ""),
                "memories": state.get("memories", []),
                "akshare_cache": state.setdefault("akshare_cache", {}),
            },
            debate_mode=debate_mode,
        ),
        "hold_analysis_pipeline",
    )
    if isinstance(pipeline_result, str):
        placeholder = pipeline_result
        result = {
            "fundamentals_report": placeholder,
            "news_report": placeholder,
            "bull_report": placeholder,
            "bear_report": placeholder,
            "summary_report": placeholder,
        }
    else:
        result = pipeline_result
    content = result.get("debate_report") if debate_mode else result["summary_report"]
    return {
        "messages": [AIMessage(content=str(content))],
        "fundamentals_report": result["fundamentals_report"],
        "news_report": result["news_report"],
        "bull_report": result["bull_report"],
        "bear_report": result["bear_report"],
    }


def _internal_route_supervisor(state: AgentState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "end"

    if len(tool_calls) == 1:
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


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("supervisor", call_supervisor_node)
    g.add_node("market_analyst", _internal_market_analyst_node)
    g.add_node("fundamental_analyst", _internal_fundamental_analyst_node)
    g.add_node("news_analyst", _internal_news_analyst_node)
    g.add_node("researchers", _internal_researchers_node)
    g.add_node("hold_analysis_pipeline", _internal_hold_analysis_pipeline_node)
    g.add_node("tool", call_tool_node)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        _internal_route_supervisor,
        {
            "market_analyst": "market_analyst",
            "fundamental_analyst": "fundamental_analyst",
            "news_analyst": "news_analyst",
            "researchers": "researchers",
            "hold_analysis_pipeline": "hold_analysis_pipeline",
            "tool": "tool",
            "end": END,
        }
    )

    g.add_edge("market_analyst", "supervisor")
    g.add_edge("fundamental_analyst", "supervisor")
    g.add_edge("news_analyst", "supervisor")
    g.add_edge("researchers", "supervisor")
    g.add_edge("hold_analysis_pipeline", "supervisor")
    g.add_edge("tool", "supervisor")
    return g.compile(checkpointer=checkpointer)
