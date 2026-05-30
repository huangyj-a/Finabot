from langgraph.graph import StateGraph, END
from finabot.agents.state import AgentState
from finabot.agents.nodes import call_llm_node as call_supervisor_node, call_tool_node, should_continue

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", call_supervisor_node)
    g.add_node("tool", call_tool_node)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        should_continue,
        {"tool": "tool", "end": END}
    )

    g.add_edge("tool", "supervisor")
    return g.compile()