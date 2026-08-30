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
            return json.dumps({"sample": [{"代码": "300502", "名称": "新易盛"}]}, ensure_ascii=False)
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


def test_hold_pipeline_moves_akshare_fetch_off_event_loop(monkeypatch):
    import finabot.agents.hold_pipeline as pipeline_module

    calls = {}

    async def fake_to_thread(func, *args):
        calls["func"] = func
        calls["args"] = args
        return {}

    async def fake_news(expression, cache):
        return "新闻"

    async def fake_bull(expression, context):
        return "看涨"

    async def fake_bear(expression, context):
        return "看跌"

    async def fake_summary(expression, context):
        return "最终总结"

    async def fake_fundamental(expression, cache=None):
        return "基本面解读"

    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(pipeline_module, "_internal_call_fundamental_analyst", fake_fundamental)
    monkeypatch.setattr(pipeline_module, "_internal_call_news_analyst", fake_news)
    monkeypatch.setattr(pipeline_module, "_internal_call_bull_researcher", fake_bull)
    monkeypatch.setattr(pipeline_module, "_internal_call_bear_researcher", fake_bear)
    monkeypatch.setattr(pipeline_module, "_internal_call_summary_manager", fake_summary)

    state_context = {"akshare_cache": {}}
    result = asyncio.run(
        pipeline_module.run_hold_analysis_pipeline("新易盛", state_context)
    )

    assert calls["func"] is pipeline_module.get_cached_akshare_data
    assert calls["args"] == (state_context["akshare_cache"], "新易盛")
    assert result["summary_report"] == "最终总结"


def test_hold_pipeline_node_returns_final_summary(monkeypatch):
    import finabot.graph.graph as graph_module

    async def fake_pipeline(expression, state_context=None, debate_mode=False):
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


def test_confidence_assessor_classifies_all_data_present():
    from finabot.agents.analysts.confidence_assessor import (
        _internal_assess_confidence,
        build_confidence_report,
    )

    akshare_data = {
        "stock_conclusion": '{"tool":"conclusion","rows":1}',
        "stock_valuation": '{"tool":"valuation","rows":2}',
        "stock_financial_indicators": '{"tool":"fi","rows":1}',
        "stock_fund_flow": '{"tool":"ff","rows":3}',
        "stock_research_report": '{"tool":"rr","rows":1}',
        "stock_notice": '{"tool":"notice","rows":0}',
        "stock_info": '{"tool":"info","rows":1}',
        "stock_snapshot": '{"tool":"snap","rows":1}',
        "stock_spot": '{"tool":"spot","rows":1}',
        "stock_news": '{"tool":"news","has_direct_news":true,"news_scope":"stock_direct"}',
    }

    assessment = _internal_assess_confidence(akshare_data, "贵州茅台")

    assert assessment["level"] == "高"
    assert assessment["score"] == 90
    assert len(assessment["coverage"]["covered"]) == 9
    assert len(assessment["coverage"]["failed"]) == 0
    assert "有直接个股新闻数据" in assessment["notes"]

    report_text = build_confidence_report(assessment)
    assert "置信评级：高（90/100）" in report_text
    assert "9 项有数据，0 项缺失" in report_text


def test_confidence_assessor_flags_errors_and_missing():
    from finabot.agents.analysts.confidence_assessor import (
        _internal_assess_confidence,
        build_confidence_report,
    )

    akshare_data = {
        "stock_conclusion": '{"tool":"conclusion","rows":1}',
        "stock_spot": '{"tool":"spot","rows":1}',
        "stock_news": '{"tool":"news","has_direct_news":false,"news_scope":"market_general"}',
    }

    assessment = _internal_assess_confidence(akshare_data, "test")

    assert assessment["level"] == "低"
    assert len(assessment["coverage"]["failed"]) == 7
    assert "仅有市场通用新闻" in assessment["notes"]

    report_text = build_confidence_report(assessment)
    assert "缺失工具" in report_text


def test_hold_pipeline_passes_confidence_to_summary(monkeypatch):
    import finabot.agents.hold_pipeline as pipeline_module

    captured = {}

    async def fake_to_thread(func, *args, **kwargs):
        return {}

    def fake_confidence(akshare_data, expression):
        captured["assessed"] = True
        return {"level": "高", "score": 90, "coverage": {"covered": [], "failed": []}, "news_scope": None, "notes": ""}

    def fake_build(assessment):
        return "置信度OK"

    async def fake_news(expression, cache):
        return "新闻"

    async def fake_bull(expression, ctx):
        return "看涨"

    async def fake_bear(expression, ctx):
        return "看跌"

    async def fake_summary(expression, context=None):
        captured["summary_context"] = context
        return "总结"

    monkeypatch.setattr(pipeline_module.asyncio, "to_thread", fake_to_thread)

    async def fake_fundamental(expression, cache):
        return None

    monkeypatch.setattr(pipeline_module, "_internal_call_fundamental_analyst", fake_fundamental)
    monkeypatch.setattr(pipeline_module, "_internal_call_news_analyst", fake_news)
    monkeypatch.setattr(pipeline_module, "_internal_call_bull_researcher", fake_bull)
    monkeypatch.setattr(pipeline_module, "_internal_call_bear_researcher", fake_bear)
    monkeypatch.setattr(pipeline_module, "_internal_call_summary_manager", fake_summary)
    monkeypatch.setattr(pipeline_module, "_internal_assess_confidence", fake_confidence)
    monkeypatch.setattr(pipeline_module, "build_confidence_report", fake_build)

    result = asyncio.run(pipeline_module.run_hold_analysis_pipeline("茅台", {"akshare_cache": {}}))

    assert captured.get("assessed")
    assert captured["summary_context"]["confidence_report"] == "置信度OK"
    assert result["confidence_report"] == "置信度OK"
    assert result["news_report"] == "新闻"


def test_hold_pipeline_debate_mode_builds_debate_report(monkeypatch):
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

    plain = asyncio.run(pipeline_module.run_hold_analysis_pipeline("茅台", {"akshare_cache": {}}))
    assert "debate_report" not in plain

    debated = asyncio.run(
        pipeline_module.run_hold_analysis_pipeline("茅台", {"akshare_cache": {}}, debate_mode=True)
    )
    assert "### 新闻分析" in debated["debate_report"]
    assert "### 看涨论点" in debated["debate_report"]
    assert "### 看跌论点" in debated["debate_report"]
    assert "### 综合结论" in debated["debate_report"]


def test_hold_pipeline_node_surfaces_debate_report_when_requested(monkeypatch):
    import finabot.graph.graph as graph_module
    from langchain_core.messages import AIMessage, HumanMessage

    async def fake_pipeline(expression, state_context=None, debate_mode=False):
        return {
            "fundamentals_report": "基础",
            "news_report": "新闻",
            "bull_report": "看涨",
            "bear_report": "看跌",
            "summary_report": "最终总结",
            "debate_report": "分步稿件",
        }

    monkeypatch.setattr(graph_module, "run_hold_analysis_pipeline", fake_pipeline)
    state = {
        "messages": [
            HumanMessage(content="未来三个月是否适合持有新易盛"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "hold_analysis_pipeline",
                        "args": {"expression": "新易盛", "debate_mode": True},
                        "id": "h1",
                    }
                ],
            ),
        ],
        "akshare_cache": {},
    }

    result = asyncio.run(graph_module._internal_hold_analysis_pipeline_node(state))

    assert isinstance(result["messages"][0], AIMessage)
    # debate_mode 时节点应直接转述分步稿件，而非简洁结论
    assert result["messages"][0].content == "分步稿件"


def test_hold_pipeline_subgraph_emits_per_step_messages(monkeypatch):
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

    initial = {
        "expression": "茅台",
        "akshare_cache": {},
        "market_report": "",
        "memories": [],
        "fundamentals_report_raw": "",
        "confidence_report": "",
        "fundamentals_report": "",
        "news_report": "",
        "bull_report": "",
        "bear_report": "",
        "summary_report": "",
        "messages": [],
    }
    result = asyncio.run(pipeline_module._HOLD_SUBGRAPH.ainvoke(initial))

    # 每个节点把自身产出追加进 messages 通道，父图据此做分步流式透出
    contents = [getattr(m, "content", "") for m in result["messages"]]
    assert len(result["messages"]) == 6
    assert any("置信评级" in c for c in contents)  # fetch 状态
    assert "基本面解读" in contents
    assert "新闻分析" in contents
    assert "看涨论点" in contents
    assert "看跌论点" in contents
    assert "综合结论" in contents
