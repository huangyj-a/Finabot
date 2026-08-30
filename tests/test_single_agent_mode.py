"""Tests for the single-agent ablation mode (评估报告: 单 Agent 对照组)."""

from langchain_core.messages import AIMessage, HumanMessage

from finabot.agents.llm import SINGLE_AGENT_SYSTEM_PROMPT
from finabot.agents.nodes import format_tools


def test_single_agent_prompt_exists_and_lists_no_sub_agents():
    assert "SINGLE_AGENT_SYSTEM_PROMPT" in dir(__import__("finabot.agents.llm", fromlist=["SINGLE_AGENT_SYSTEM_PROMPT"]))
    assert "hold_analysis_pipeline" not in SINGLE_AGENT_SYSTEM_PROMPT
    assert "market_analyst" not in SINGLE_AGENT_SYSTEM_PROMPT
    assert "东方财富" in SINGLE_AGENT_SYSTEM_PROMPT  # 保留统一引用规范


def test_format_tools_single_agent_excludes_sub_agents():
    sub_agents = {"market_analyst", "fundamental_analyst", "news_analyst", "researchers", "hold_analysis_pipeline"}
    multi_names = {t["function"]["name"] for t in format_tools(single_agent=False)}
    single_names = {t["function"]["name"] for t in format_tools(single_agent=True)}

    # 多 Agent 模式含子代理；单 Agent 模式剔除子代理但仍保留数据工具
    assert sub_agents <= multi_names
    assert sub_agents.isdisjoint(single_names)
    assert "stock_a_lookup" in single_names
    assert "calculator" in single_names


def test_build_graph_single_agent_compiles():
    from finabot.graph.graph import build_graph

    graph = build_graph(single_agent=True)
    assert graph is not None
    # 单 Agent 图节点：supervisor + tool（无子代理节点）
    nodes = graph.get_graph().nodes
    assert "supervisor" in nodes
    assert "tool" in nodes
    assert "market_analyst" not in nodes
    assert "hold_analysis_pipeline" not in nodes


def test_build_graph_multi_agent_keeps_sub_agent_nodes():
    from finabot.graph.graph import build_graph

    nodes = build_graph(single_agent=False).get_graph().nodes
    assert "market_analyst" in nodes
    assert "hold_analysis_pipeline" in nodes


def test_single_agent_route_never_returns_sub_agent():
    from finabot.graph.graph import _internal_make_route_supervisor

    route = _internal_make_route_supervisor(single_agent=True)
    state = {
        "messages": [
            HumanMessage(content="分析"),
            AIMessage(content="", tool_calls=[{"name": "hold_analysis_pipeline", "args": {}, "id": "x"}]),
        ]
    }
    # 单 Agent 模式：即便出现子代理 tool_call，也落到通用 tool 节点而非子代理分支
    assert route(state) == "tool"


def test_multi_agent_route_returns_sub_agent():
    from finabot.graph.graph import _internal_make_route_supervisor

    route = _internal_make_route_supervisor(single_agent=False)
    state = {
        "messages": [
            HumanMessage(content="分析"),
            AIMessage(content="", tool_calls=[{"name": "hold_analysis_pipeline", "args": {}, "id": "x"}]),
        ]
    }
    assert route(state) == "hold_analysis_pipeline"