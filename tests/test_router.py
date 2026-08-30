"""Tests for the deterministic intent pre-router (finabot.graph.router)."""

from langchain_core.messages import HumanMessage

from finabot.graph.router import classify_intent


def test_hold_question_with_stock_code_short_circuits():
    target, debate = classify_intent("600519 现在适合持有吗")
    assert target == "hold_analysis_pipeline"
    assert debate is False


def test_hold_question_with_suffixed_company_name_short_circuits():
    target, _ = classify_intent("贵州茅台股份适合持有吗")
    assert target == "hold_analysis_pipeline"


def test_hold_question_with_famous_bare_name_short_circuits():
    target, _ = classify_intent("贵州茅台现在适合持有吗")
    assert target == "hold_analysis_pipeline"


def test_hold_question_with_debate_keywords_sets_debate_mode():
    target, debate = classify_intent("茅台适合持有吗？请分别展开多空辩论")
    assert target == "hold_analysis_pipeline"
    assert debate is True


def test_generic_stock_buying_question_does_not_short_circuit():
    # "买股票" 没有具体标的（股票不是公司后缀/知名名锚点），应回落 LLM
    target, debate = classify_intent("买股票怎么样")
    assert target is None
    assert debate is False


def test_generic_hold_intent_without_stock_target_falls_back():
    target, _ = classify_intent("该不该买股票")
    assert target is None


def test_market_level_question_short_circuits_to_market_analyst():
    target, _ = classify_intent("大盘走势怎么样")
    assert target == "market_analyst"


def test_single_stock_price_question_does_not_go_to_market_analyst():
    # 点名个股的数据问题（"茅台涨了多少"）不应错误送进市场分析
    target, _ = classify_intent("茅台今天涨了多少")
    assert target is None


def test_fundamental_research_question_falls_back():
    target, _ = classify_intent("茅台的基本面怎么样")
    assert target is None


def test_compliance_boundary_question_never_short_circuits():
    # 具体买卖/仓位请求必须走 LLM 路径以触发合规提示
    target, _ = classify_intent("买入茅台股份多少仓位")
    assert target is None


def test_compliance_caution_question_never_short_circuits():
    # 带研究限定的买卖问题（caution 级）同样不短路
    target, _ = classify_intent("茅台加仓策略分析")
    assert target is None


def test_empty_question_falls_back():
    target, debate = classify_intent("")
    assert target is None
    assert debate is False


def test_multi_turn_followup_without_target_falls_back():
    # "那它呢" 依赖上下文，规则无法判断，必须交给 LLM
    target, _ = classify_intent("那它呢，还能继续拿吗")
    assert target is None


def test_graph_wires_router_before_supervisor():
    import finabot.graph.graph as graph_module

    graph = graph_module.build_graph()
    assert "router" in graph.nodes
    assert "__start__" in graph.nodes
    assert "supervisor" in graph.nodes
    assert "hold_analysis_pipeline" in graph.nodes


def test_single_agent_graph_has_no_router():
    import finabot.graph.graph as graph_module

    graph = graph_module.build_graph(single_agent=True)
    assert "router" not in graph.nodes


def test_router_node_writes_debate_mode_when_debate_intent():
    import finabot.graph.graph as graph_module

    state = {"messages": [HumanMessage(content="茅台适合持有吗？分别看多空")]}
    result = graph_module._internal_router_node(state)
    assert result.get("debate_mode") is True

    state2 = {"messages": [HumanMessage(content="600519 还能买吗")]}
    result2 = graph_module._internal_router_node(state2)
    assert result2 == {"debate_mode": False}


def test_route_intent_falls_back_to_supervisor():
    import finabot.graph.graph as graph_module

    state = {"messages": [HumanMessage(content="你好")]}
    assert graph_module._internal_route_intent(state) == "supervisor"

    hold_state = {"messages": [HumanMessage(content="新易盛现在适合持有吗")]}
    assert graph_module._internal_route_intent(hold_state) == "hold_analysis_pipeline"
