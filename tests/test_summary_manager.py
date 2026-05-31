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


def test_graph_summary_context_uses_reports(monkeypatch):
    import finabot.graph.graph as graph_module

    async def fake_summary(expression: str, context: dict | None = None) -> str:
        assert context["market_report"] == "市场"
        assert context["news_report"] == "新闻"
        assert context["bull_report"] == "看涨"
        assert context["bear_report"] == "看跌"
        return "总结结论"

    monkeypatch.setattr(graph_module, "_internal_call_summary_manager", fake_summary)
    state = {
        "messages": [HumanMessage(content="总结一下")],
        "market_report": "市场",
        "news_report": "新闻",
        "bull_report": "看涨",
        "bear_report": "看跌",
    }

    result = __import__("asyncio").run(graph_module._internal_summary_manager_node(state))

    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "总结结论"


def test_tools_include_summary_manager():
    sys.modules.setdefault("akshare", types.ModuleType("akshare"))
    from finabot.tools.base import get_tools

    assert "summary_manager" in [tool.name for tool in get_tools()]
