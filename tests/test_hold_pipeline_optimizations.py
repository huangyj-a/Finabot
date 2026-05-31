import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage


class FakeTool:
    def __init__(self, name: str, calls: dict[str, int]):
        self.name = name
        self.calls = calls

    def invoke(self, kwargs):
        self.calls[self.name] = self.calls.get(self.name, 0) + 1
        if self.name == "lookup":
            return json.dumps({"candidates": [{"代码": "300502", "名称": "新易盛"}]}, ensure_ascii=False)
        return f"{self.name}:{kwargs}"


def test_akshare_cache_reuses_same_stock_fetches(monkeypatch):
    import finabot.agents.akshare_cache as cache_module
    import finabot.tools.akshare_tools as akshare_tools

    calls = {}
    monkeypatch.setattr(akshare_tools, "stock_a_lookup", FakeTool("lookup", calls))
    monkeypatch.setattr(akshare_tools, "stock_a_spot", FakeTool("spot", calls))
    monkeypatch.setattr(akshare_tools, "stock_a_individual_info", FakeTool("info", calls))
    monkeypatch.setattr(akshare_tools, "stock_a_snapshot", FakeTool("snapshot", calls))
    monkeypatch.setattr(akshare_tools, "stock_a_conclusion", FakeTool("conclusion", calls))

    shared_cache = {}
    first = cache_module.get_cached_akshare_data(shared_cache, "新易盛")
    second = cache_module.get_cached_akshare_data(shared_cache, "新易盛")

    assert first is second
    assert calls == {"lookup": 1, "spot": 1, "info": 1, "snapshot": 1, "conclusion": 1}


def test_hold_pipeline_node_returns_final_summary(monkeypatch):
    import finabot.graph.graph as graph_module

    async def fake_pipeline(expression, state_context=None):
        assert state_context["akshare_cache"] == {}
        return {
            "fundamentals_report": "基础",
            "news_report": "新闻",
            "bull_report": "看涨",
            "bear_report": "看跌",
            "summary_report": "最终总结",
        }

    monkeypatch.setattr(graph_module, "run_hold_analysis_pipeline", fake_pipeline)
    state = {"messages": [HumanMessage(content="未来三个月是否适合持有新易盛")], "akshare_cache": {}}

    result = asyncio.run(graph_module._internal_hold_analysis_pipeline_node(state))

    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "最终总结"
    assert result["news_report"] == "新闻"
