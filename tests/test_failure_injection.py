"""Tests for eval failure injection and the no-bear ablation."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from finabot.agents.failure import injected_failure


def test_injected_failure_unset_returns_none(monkeypatch):
    monkeypatch.delenv("FINABOT_EVAL_FAIL_NODE", raising=False)
    assert injected_failure("news_analyst") is None


def test_injected_failure_hits_specific_node(monkeypatch):
    monkeypatch.setenv("FINABOT_EVAL_FAIL_NODE", "news_analyst")
    assert "eval_failure:news_analyst" in injected_failure("news_analyst")
    assert injected_failure("market_analyst") is None


def test_injected_failure_all(monkeypatch):
    monkeypatch.setenv("FINABOT_EVAL_FAIL_NODE", "all")
    assert injected_failure("any_node") is not None


def test_news_analyst_node_injects_failure(monkeypatch):
    import finabot.graph.graph as graph_module

    monkeypatch.setenv("FINABOT_EVAL_FAIL_NODE", "news_analyst")
    state = {"messages": [HumanMessage(content="分析新闻")], "risk_flags": []}
    result = asyncio.run(graph_module._internal_news_analyst_node(state))

    assert "eval_failure:news_analyst" in result["messages"][0].content
    assert result["news_report"].startswith("[eval_failure")
    # 失败注入应记入 risk_flags（"最高风险不消失"不变量）
    assert any("eval_failure" in flag for flag in result["risk_flags"])


def test_no_bear_subgraph_builds_without_bear():
    from finabot.agents.hold_pipeline import build_hold_analysis_subgraph

    with_bear = build_hold_analysis_subgraph(include_bear=True).get_graph().nodes
    without_bear = build_hold_analysis_subgraph(include_bear=False).get_graph().nodes

    assert "bear" in with_bear
    assert "bear" not in without_bear
    assert "bull" in without_bear  # 看涨仍保留


def test_no_bear_pipeline_returns_empty_bear(monkeypatch):
    import finabot.agents.hold_pipeline as pipeline_module

    async def fake_to_thread(func, *args, **kwargs):
        return {}

    async def fake_fundamental(expression, cache=None):
        return "基本面解读"

    async def fake_news(expression, cache=None):
        return "新闻分析"

    async def fake_bull(expression, ctx):
        return "看涨论点"

    async def fake_summary(expression, context=None):
        return "综合结论"

    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(pipeline_module, "_internal_call_fundamental_analyst", fake_fundamental)
    monkeypatch.setattr(pipeline_module, "_internal_call_news_analyst", fake_news)
    monkeypatch.setattr(pipeline_module, "_internal_call_bull_researcher", fake_bull)
    monkeypatch.setattr(pipeline_module, "_internal_call_summary_manager", fake_summary)

    result = asyncio.run(
        pipeline_module.run_hold_analysis_pipeline("茅台", {"akshare_cache": {}}, include_bear=False)
    )

    assert result["bear_report"] == ""
    assert result["bull_report"] == "看涨论点"
    assert result["summary_report"] == "综合结论"


def test_pipeline_returns_three_way_evidence(monkeypatch):
    import finabot.agents.hold_pipeline as pipeline_module

    async def fake_to_thread(func, *args, **kwargs):
        return {}

    async def fake_fundamental(expression, cache=None):
        return "基本面解读"

    async def fake_news(expression, cache=None):
        return "新闻分析"

    async def fake_bull(expression, ctx):
        return "看涨论点"

    async def fake_bear(expression, ctx):
        return "看跌论点"

    async def fake_summary(expression, context=None):
        return "综合结论"

    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(pipeline_module, "_internal_call_fundamental_analyst", fake_fundamental)
    monkeypatch.setattr(pipeline_module, "_internal_call_news_analyst", fake_news)
    monkeypatch.setattr(pipeline_module, "_internal_call_bull_researcher", fake_bull)
    monkeypatch.setattr(pipeline_module, "_internal_call_bear_researcher", fake_bear)
    monkeypatch.setattr(pipeline_module, "_internal_call_summary_manager", fake_summary)

    result = asyncio.run(
        pipeline_module.run_hold_analysis_pipeline("茅台", {"akshare_cache": {}})
    )

    # 支持/反对证据分别来自看涨/看跌论点（结构化关闭时以自由文本形式记录）
    assert "看涨论点" in result["supporting_evidence"]
    assert "看跌论点" in result["opposing_evidence"]