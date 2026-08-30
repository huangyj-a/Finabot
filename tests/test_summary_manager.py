import sys
import types

from langchain_core.messages import AIMessage, HumanMessage


def test_summary_input_includes_all_analysis_sections(monkeypatch):
    from finabot.agents.managers import manager

    monkeypatch.setattr(manager, "_internal_collect_fundamental_context", lambda expression, cache=None: "基本面数据")
    content = manager._internal_format_summary_input(
        "贵州茅台适合持有吗",
        {
            "market_report": "市场分析数据",
            "news_report": "新闻分析数据",
            "bull_report": "看涨数据",
            "bear_report": "看跌数据",
            "memories": [{"content": "用户偏稳健"}],
        },
    )

    assert "基本面数据" in content
    assert "市场分析数据" in content
    assert "新闻分析数据" in content
    assert "看涨数据" in content
    assert "看跌数据" in content
    assert "用户偏稳健" in content


def test_summary_input_echoes_risk_flags(monkeypatch):
    from finabot.agents.managers import manager

    monkeypatch.setattr(manager, "_internal_collect_fundamental_context", lambda expression, cache=None: "基本面数据")
    content = manager._internal_format_summary_input(
        "贵州茅台适合持有吗",
        {"risk_flags": ["估值过高", "批价波动"]},
    )

    assert "最高风险清单" in content
    assert "估值过高" in content
    assert "批价波动" in content
    assert "不得无解释消失" in content


def test_summary_prompt_requires_risk_echo():
    from finabot.agents.managers.manager import _SUMMARY_MANAGER_PROMPT
    assert "最高风险回显" in _SUMMARY_MANAGER_PROMPT


def test_summary_input_includes_three_way_evidence(monkeypatch):
    from finabot.agents.managers import manager

    monkeypatch.setattr(manager, "_internal_collect_fundamental_context", lambda expression, cache=None: "基本面数据")
    content = manager._internal_format_summary_input(
        "贵州茅台适合持有吗",
        {
            "supporting_evidence": ["品牌壁垒强"],
            "opposing_evidence": ["估值不低"],
            "unknown_evidence": ["批价数据缺失"],
        },
    )

    assert "支持/反对/未知证据" in content
    assert "品牌壁垒强" in content
    assert "估值不低" in content
    assert "批价数据缺失" in content
    assert "冲突不得丢失" in content


def test_hold_pipeline_node_writes_reports_back_to_state(monkeypatch):
    import asyncio

    import finabot.graph.graph as graph_module

    async def fake_pipeline(expression, state_context=None, debate_mode=False):
        assert expression == "贵州茅台适合持有吗"
        return {
            "fundamentals_report": "基本面",
            "news_report": "新闻",
            "bull_report": "看涨",
            "bear_report": "看跌",
            "summary_report": "总结结论",
        }

    monkeypatch.setattr(graph_module, "run_hold_analysis_pipeline", fake_pipeline)
    state = {
        "messages": [HumanMessage(content="贵州茅台适合持有吗")],
        "market_report": "市场",
    }

    result = asyncio.run(graph_module._internal_hold_analysis_pipeline_node(state))

    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "总结结论"
    # 流水线把各阶段报告写回 state，供后续 supervisor/会话复用
    assert result["news_report"] == "新闻"
    assert result["bull_report"] == "看涨"
    assert result["bear_report"] == "看跌"
    assert result["fundamentals_report"] == "基本面"


def test_tools_include_hold_pipeline_and_drop_folded_agents():
    sys.modules.setdefault("akshare", types.ModuleType("akshare"))
    from finabot.tools.base import get_tools

    tool_names = [tool.name for tool in get_tools()]
    # 折叠后：单股分析统一走 hold_analysis_pipeline
    assert "hold_analysis_pipeline" in tool_names
    # bull/bear/summary_manager 已折叠进流水线，不再作为 supervisor 可调用的独立子代理
    assert "bull_researcher" not in tool_names
    assert "bear_researcher" not in tool_names
    assert "summary_manager" not in tool_names


def test_summary_manager_fetches_fundamentals_off_event_loop(monkeypatch):
    import asyncio

    from finabot.agents.managers import manager


    calls = {}

    def fake_collect(expression, cache=None):
        calls["collect_args"] = (expression, cache)
        return "基本面数据"

    async def fake_to_thread(func, *args, **kwargs):
        calls["thread_func"] = func
        return func(*args, **kwargs)

    async def fake_glm_call(messages=None, memories=None, tools=None, stream_label=None):
        calls["prompt_content"] = messages[-1].content
        return types.SimpleNamespace(content="总结结论")


    monkeypatch.setattr(manager.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(manager, "_internal_collect_fundamental_context", fake_collect)
    monkeypatch.setattr(manager, "litellm_glm_call", fake_glm_call)


    reply = asyncio.run(
        manager._internal_call_summary_manager("贵州茅台", {"market_report": "市场"})
    )


    assert reply == "总结结论"
    # 基本面抓取必须经 to_thread 执行，避免阻塞事件循环
    assert calls["thread_func"] is fake_collect
    assert calls["collect_args"] == ("贵州茅台", None)
    assert "基本面数据" in calls["prompt_content"]
    assert "市场分析数据" in calls["prompt_content"]
