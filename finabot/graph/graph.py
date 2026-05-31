from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from finabot.agents.state import AgentState
from finabot.agents.analysts import _internal_call_news_analyst, market_analyst
from finabot.agents.hold_pipeline import run_hold_analysis_pipeline
from finabot.agents.managers import _internal_call_summary_manager
from finabot.agents.nodes import call_llm_node as call_supervisor_node, call_tool_node
from finabot.agents.researchers import (
    _internal_call_bear_researcher,
    _internal_call_bull_researcher,
    researchers,
)


def _internal_latest_user_message(state: AgentState) -> str:
    messages = state.get("messages", []) or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            content = getattr(message, "content", "") or ""
            if content:
                return str(content)
    if messages:
        return str(getattr(messages[-1], "content", "") or "")
    return ""


async def _internal_market_analyst_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await market_analyst.ainvoke({"expression": expression})
    return {"messages": [AIMessage(content=str(result))], "market_report": str(result)}


async def _internal_news_analyst_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await _internal_call_news_analyst(expression, state.setdefault("akshare_cache", {}))
    return {"messages": [AIMessage(content=str(result))], "news_report": str(result)}


async def _internal_researchers_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await researchers.ainvoke({"expression": expression})
    return {"messages": [AIMessage(content=str(result))]}


def _internal_get_debate_context(state: AgentState) -> dict:
    debate_context = dict(state.get("debate_context", {}) or {})
    debate_context.setdefault("history", "")
    debate_context.setdefault("bull_history", "")
    debate_context.setdefault("bear_history", "")
    debate_context.setdefault("current_response", "")
    debate_context.setdefault("last_bull_argument", "")
    debate_context.setdefault("last_bear_argument", "")
    debate_context.setdefault("last_speaker", None)
    debate_context.setdefault("count", 0)
    return debate_context


def _internal_record_debate_argument(debate_context: dict, speaker: str, content: str) -> dict:
    updated = dict(debate_context)
    label = "Bull Analyst" if speaker == "bull" else "Bear Analyst"
    argument = f"{label}: {content}"
    history = updated.get("history", "")
    speaker_history_key = "bull_history" if speaker == "bull" else "bear_history"
    last_argument_key = "last_bull_argument" if speaker == "bull" else "last_bear_argument"

    updated["history"] = f"{history}\n{argument}".strip()
    updated[speaker_history_key] = f"{updated.get(speaker_history_key, '')}\n{argument}".strip()
    updated[last_argument_key] = argument
    updated["current_response"] = argument
    updated["last_speaker"] = speaker
    updated["count"] = int(updated.get("count", 0) or 0) + 1
    updated["in_progress"] = None
    return updated


def _internal_build_summary_context(state: AgentState) -> dict:
    debate_context = _internal_get_debate_context(state)
    return {
        "market_report": state.get("market_report", ""),
        "news_report": state.get("news_report", ""),
        "bull_report": state.get("bull_report", "") or debate_context.get("last_bull_argument", ""),
        "bear_report": state.get("bear_report", "") or debate_context.get("last_bear_argument", ""),
        "fundamentals_report": state.get("fundamentals_report", ""),
        "memories": state.get("memories", []),
        "akshare_cache": state.get("akshare_cache", {}),
    }


async def _internal_bull_researcher_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    debate_context = _internal_get_debate_context(state)
    if state.get("news_report"):
        debate_context["news_report"] = state.get("news_report")
    result = await _internal_call_bull_researcher(expression, debate_context)
    updated_context = _internal_record_debate_argument(debate_context, "bull", str(result))
    return {"messages": [AIMessage(content=str(result))], "debate_context": updated_context, "bull_report": str(result)}


async def _internal_bear_researcher_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    debate_context = _internal_get_debate_context(state)
    if state.get("news_report"):
        debate_context["news_report"] = state.get("news_report")
    result = await _internal_call_bear_researcher(expression, debate_context)
    updated_context = _internal_record_debate_argument(debate_context, "bear", str(result))
    return {"messages": [AIMessage(content=str(result))], "debate_context": updated_context, "bear_report": str(result)}


async def _internal_summary_manager_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await _internal_call_summary_manager(expression, _internal_build_summary_context(state))
    return {"messages": [AIMessage(content=str(result))]}


async def _internal_hold_analysis_pipeline_node(state: AgentState):
    expression = _internal_latest_user_message(state)
    result = await run_hold_analysis_pipeline(
        expression,
        {
            "market_report": state.get("market_report", ""),
            "memories": state.get("memories", []),
            "akshare_cache": state.setdefault("akshare_cache", {}),
        },
    )
    return {
        "messages": [AIMessage(content=result["summary_report"])],
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
        if tool_name == "news_analyst":
            return "news_analyst"
        if tool_name == "researchers":
            return "researchers"
        if tool_name == "bull_researcher":
            return "bull_researcher"
        if tool_name == "bear_researcher":
            return "bear_researcher"
        if tool_name == "summary_manager":
            return "summary_manager"
        if tool_name == "hold_analysis_pipeline":
            return "hold_analysis_pipeline"

    return "tool"

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", call_supervisor_node)
    g.add_node("market_analyst", _internal_market_analyst_node)
    g.add_node("news_analyst", _internal_news_analyst_node)
    g.add_node("researchers", _internal_researchers_node)
    g.add_node("bull_researcher", _internal_bull_researcher_node)
    g.add_node("bear_researcher", _internal_bear_researcher_node)
    g.add_node("summary_manager", _internal_summary_manager_node)
    g.add_node("hold_analysis_pipeline", _internal_hold_analysis_pipeline_node)
    g.add_node("tool", call_tool_node)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        _internal_route_supervisor,
        {
            "market_analyst": "market_analyst",
            "news_analyst": "news_analyst",
            "researchers": "researchers",
            "bull_researcher": "bull_researcher",
            "bear_researcher": "bear_researcher",
            "summary_manager": "summary_manager",
            "hold_analysis_pipeline": "hold_analysis_pipeline",
            "tool": "tool",
            "end": END,
        }
    )

    g.add_edge("market_analyst", "supervisor")
    g.add_edge("news_analyst", "supervisor")
    g.add_edge("researchers", "supervisor")
    g.add_edge("bull_researcher", "supervisor")
    g.add_edge("bear_researcher", "supervisor")
    g.add_edge("summary_manager", "supervisor")
    g.add_edge("hold_analysis_pipeline", "supervisor")
    g.add_edge("tool", "supervisor")
    return g.compile()
