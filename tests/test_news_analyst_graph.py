import sys
import types

from langchain_core.messages import AIMessage, HumanMessage


def test_bull_and_bear_prompt_include_news_report():
    from finabot.agents.researchers.bear_researcher import _internal_format_expression as bear_format
    from finabot.agents.researchers.bull_researcher import _internal_format_expression as bull_format

    debate_context = {"news_report": "新闻显示订单增长，但监管风险上升"}

    assert "新闻显示订单增长" in bull_format("分析某股票", debate_context)
    assert "新闻显示订单增长" in bear_format("分析某股票", debate_context)


def test_graph_routes_news_analyst_and_persists_report(monkeypatch):
    import finabot.graph.graph as graph_module

    async def fake_news(expression: str, cache=None) -> str:
        return f"新闻报告：{expression}"

    monkeypatch.setattr(graph_module, "_internal_call_news_analyst", fake_news)

    state = {"messages": [HumanMessage(content="分析贵州茅台新闻影响")]}
    result = __import__("asyncio").run(graph_module._internal_news_analyst_node(state))

    assert result["news_report"] == "新闻报告：分析贵州茅台新闻影响"
    assert isinstance(result["messages"][0], AIMessage)


def test_tools_include_news_analyst(monkeypatch):
    sys.modules.setdefault("akshare", types.ModuleType("akshare"))
    from finabot.tools.base import get_tools

    assert "news_analyst" in [tool.name for tool in get_tools()]
