import asyncio
from types import SimpleNamespace

import pytest

from finabot.bus.events import InboundMessage
from finabot.bus.queue import MessageBus


def _message(chat_id: str, content: str) -> InboundMessage:
    return InboundMessage(
        channel="cli",
        sender_id="tester",
        chat_id=chat_id,
        content=content,
    )


@pytest.mark.anyio
async def test_same_session_messages_are_serialized():
    from finabot.agents.core import Agent

    agent = object.__new__(Agent)
    agent.bus = MessageBus()
    agent._session_locks = {}
    active = 0
    maximum_active = 0

    async def fake_process_locked(msg):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1

    agent._process_locked = fake_process_locked

    await asyncio.gather(
        agent.process(_message("same", "one")),
        agent.process(_message("same", "two")),
    )

    assert maximum_active == 1


@pytest.mark.anyio
async def test_different_sessions_can_run_concurrently():
    from finabot.agents.core import Agent

    agent = object.__new__(Agent)
    agent.bus = MessageBus()
    agent._session_locks = {}
    active = 0
    maximum_active = 0
    both_started = asyncio.Event()

    async def fake_process_locked(msg):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        active -= 1

    agent._process_locked = fake_process_locked

    await asyncio.gather(
        agent.process(_message("one", "one")),
        agent.process(_message("two", "two")),
    )

    assert maximum_active == 2


@pytest.mark.anyio
async def test_agent_exception_always_publishes_error_response():
    from finabot.agents.core import Agent

    agent = object.__new__(Agent)
    agent.bus = MessageBus()
    agent._session_locks = {}

    async def fail(msg):
        raise RuntimeError("private failure detail")

    agent._process_locked = fail
    await agent.process(_message("error", "boom"))
    outbound = await asyncio.wait_for(agent.bus.consume_outbound(), timeout=0.1)

    assert outbound.content == "Finabot 处理请求失败，请稍后重试。"
    assert outbound.metadata == {"error": True, "error_type": "RuntimeError"}
    assert "private failure detail" not in outbound.content


def test_run_scoped_state_is_cleared_between_turns():
    from finabot.agents.core import Agent

    agent = object.__new__(Agent)
    state = {
        "messages": ["preserved"],
        "market_report": "old market",
        "news_report": "old news",
        "bull_report": "old bull",
        "bear_report": "old bear",
        "fundamentals_report": "old fundamentals",
        "debate_context": {"old": True},
        "akshare_cache": {"old": True},
    }

    agent._reset_run_state(state)

    assert state["messages"] == ["preserved"]
    assert state["market_report"] == ""
    assert state["news_report"] == ""
    assert state["bull_report"] == ""
    assert state["bear_report"] == ""
    assert state["fundamentals_report"] == ""
    assert state["debate_context"] == {}
    assert state["akshare_cache"] == {}


def test_cli_response_timeout_requires_positive_number(monkeypatch):
    import typer

    from finabot.cli.commands import _response_timeout_seconds

    monkeypatch.setenv("FINABOT_RESPONSE_TIMEOUT_SECONDS", "invalid")
    with pytest.raises(typer.BadParameter, match="必须是正数"):
        _response_timeout_seconds()

    monkeypatch.setenv("FINABOT_RESPONSE_TIMEOUT_SECONDS", "0")
    with pytest.raises(typer.BadParameter, match="必须是正数"):
        _response_timeout_seconds()

    monkeypatch.setenv("FINABOT_RESPONSE_TIMEOUT_SECONDS", "15.5")
    assert _response_timeout_seconds() == 15.5


def test_llm_settings_are_read_at_call_time(monkeypatch):
    from finabot.agents.llm import get_llm_settings

    # litellm 在 import 时会对 CWD 的 .env 执行 load_dotenv()，
    # 这里必须清掉通用覆盖变量，保证用例只受自身注入的环境影响。
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("LLM_MODEL", "glm-test")
    monkeypatch.setenv("ZAI_API_KEY", "first")
    first = get_llm_settings()

    monkeypatch.setenv("ZAI_API_KEY", "second")
    second = get_llm_settings()

    assert first.api_key == "first"
    assert second.api_key == "second"
    assert second.litellm_model == "zai/glm-test"


def test_llm_settings_support_openai_compatible_endpoint(monkeypatch):
    from finabot.agents.llm import get_llm_settings

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "inferai-secret")
    monkeypatch.setenv("LLM_API_BASE", "https://inferaichat.com/v1")

    settings = get_llm_settings()

    assert settings.provider == "openai"
    assert settings.model == "deepseek-v4-pro"
    assert settings.litellm_model == "openai/deepseek-v4-pro"
    assert settings.api_key == "inferai-secret"
    assert settings.api_base == "https://inferaichat.com/v1"


@pytest.mark.anyio
async def test_internal_acompletion_forwards_custom_api_base(monkeypatch):
    import finabot.agents.llm as llm_module

    captured_kwargs = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    settings = llm_module.LLMSettings(
        provider="openai",
        model="deepseek-v4-pro",
        litellm_model="openai/deepseek-v4-pro",
        api_key="secret",
        timeout_seconds=1,
        api_base="https://inferaichat.com/v1",
    )

    await llm_module._internal_acompletion(
        settings,
        [{"role": "user", "content": "hi"}],
        None,
        retry=False,
    )

    assert captured_kwargs["model"] == "openai/deepseek-v4-pro"
    assert captured_kwargs["api_key"] == "secret"
    assert captured_kwargs["api_base"] == "https://inferaichat.com/v1"


@pytest.mark.anyio
async def test_tool_node_appends_error_tool_message_for_unknown_tool(monkeypatch):
    from langchain_core.messages import AIMessage

    from finabot.agents import nodes as nodes_module

    class FakeCalculatorTool:
        name = "calculator"
        description = "calc"
        args_schema = None

        async def ainvoke(self, args):
            return "42"

    monkeypatch.setattr(nodes_module, "tools", [FakeCalculatorTool()])
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"},
                    {"name": "does_not_exist", "args": {}, "id": "c2"},
                ],
            )
        ]
    }

    result = await nodes_module.call_tool_node(state)

    # 每个 tool_call 都必须有对应 ToolMessage，否则下一条请求会被 API 拒绝
    assert len(result["messages"]) == 2
    assert result["messages"][0].tool_call_id == "c1"
    assert result["messages"][0].content == "42"
    assert result["messages"][1].tool_call_id == "c2"
    assert "未知工具" in result["messages"][1].content


@pytest.mark.anyio
async def test_call_llm_node_backfills_missing_tool_call_ids(monkeypatch):
    from types import SimpleNamespace

    from finabot.agents import nodes as nodes_module

    async def fake_glm_call(messages=None, tools=None, memories=None, stream_label=None):
        return SimpleNamespace(
            content="",
            tool_calls=[
                {"function": {"name": "calculator", "arguments": "{}"}, "id": None},
                {"function": {"name": "read_file", "arguments": "{}"}, "id": None},
            ],
        )

    monkeypatch.setattr(nodes_module, "litellm_glm_call", fake_glm_call)

    result = await nodes_module.call_llm_node({"messages": [], "memories": []})

    tool_calls = result["messages"][0].tool_calls
    assert [call["id"] for call in tool_calls] == ["finabot_call_0", "finabot_call_1"]


def _internal_state_with_sub_agent_call(tool_name: str, call_id: str) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage

    return {
        "messages": [
            HumanMessage(content="贵州茅台适合持有吗"),
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {"expression": "ignored"}, "id": call_id}],
            ),
        ],
        "market_report": "市场报告",
        "news_report": "新闻报告",
        "bull_report": "",
        "bear_report": "",
        "fundamentals_report": "基础数据",
        "memories": [{"content": "稳健型"}],
        "akshare_cache": {"600519": {"cached": True}},
        "debate_context": {"count": 1},
    }


@pytest.mark.anyio
async def test_tool_route_feeds_hold_pipeline_same_context_as_node_route(monkeypatch):
    from finabot.agents import nodes as nodes_module

    captured = {}

    async def fake_pipeline(expression, state_context=None, debate_mode=False):
        captured["expression"] = expression
        captured["context"] = state_context
        return {
            "fundamentals_report": "基本面",
            "news_report": "新闻",
            "bull_report": "看涨",
            "bear_report": "看跌",
            "summary_report": "总结结论",
        }

    monkeypatch.setattr(nodes_module, "run_hold_analysis_pipeline", fake_pipeline)

    state = _internal_state_with_sub_agent_call("hold_analysis_pipeline", "h1")
    result = await nodes_module.call_tool_node(state)

    # 与节点路由一致：消费最新人类消息而非工具参数，且共享 AKShare 缓存
    assert captured["expression"] == "贵州茅台适合持有吗"
    context = captured["context"]
    assert context["market_report"] == "市场报告"
    assert context["akshare_cache"] is state["akshare_cache"]

    assert len(result["messages"]) == 1
    assert result["messages"][0].tool_call_id == "h1"
    assert result["messages"][0].content == "总结结论"
    # 报告写回 state，与节点路由行为一致
    assert result["news_report"] == "新闻"
    assert result["bull_report"] == "看涨"
    assert result["bear_report"] == "看跌"
    assert result["fundamentals_report"] == "基本面"


@pytest.mark.anyio
async def test_tool_route_hold_pipeline_shares_cache_and_writes_reports(monkeypatch):
    from finabot.agents import nodes as nodes_module

    captured = {}

    async def fake_pipeline(expression, state_context=None, debate_mode=False):
        captured["cache"] = state_context["akshare_cache"]
        return {
            "fundamentals_report": "基本面",
            "news_report": "新闻",
            "bull_report": "看涨",
            "bear_report": "看跌",
            "summary_report": "总结",
        }

    monkeypatch.setattr(nodes_module, "run_hold_analysis_pipeline", fake_pipeline)

    state = _internal_state_with_sub_agent_call("hold_analysis_pipeline", "h1")
    result = await nodes_module.call_tool_node(state)

    # 与节点路由共享同一份 AKShare 缓存（流水线内部据此去重抓取）
    assert captured["cache"] is state["akshare_cache"]
    # 多空与新闻报告写回，后续环节（如再次调用流水线）可直接复用
    assert result["news_report"] == "新闻"
    assert result["bull_report"] == "看涨"
    assert result["bear_report"] == "看跌"


@pytest.mark.anyio
async def test_tool_route_hold_pipeline_passes_debate_mode_through(monkeypatch):
    from finabot.agents import nodes as nodes_module
    from langchain_core.messages import AIMessage, HumanMessage

    captured = {}

    async def fake_pipeline(expression, state_context=None, debate_mode=False):
        captured["debate_mode"] = debate_mode
        return {
            "fundamentals_report": "基本面",
            "news_report": "新闻",
            "bull_report": "看涨",
            "bear_report": "看跌",
            "summary_report": "简洁结论",
            "debate_report": "分步辩论稿件",
        }

    monkeypatch.setattr(nodes_module, "run_hold_analysis_pipeline", fake_pipeline)

    state = {
        "messages": [
            HumanMessage(content="贵州茅台适合持有吗"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "hold_analysis_pipeline",
                        "args": {"expression": "贵州茅台", "debate_mode": True},
                        "id": "h1",
                    }
                ],
            ),
        ],
        "akshare_cache": {"600519": {"cached": True}},
    }
    result = await nodes_module.call_tool_node(state)

    assert captured["debate_mode"] is True
    # debate_mode 时工具路由返回的是分步稿件，而非简洁结论
    assert result["messages"][0].content == "分步辩论稿件"


@pytest.mark.anyio
async def test_tool_route_mixed_calls_handle_sub_agent_and_regular_tools(monkeypatch):
    from langchain_core.messages import AIMessage

    from finabot.agents import nodes as nodes_module

    class FakeCalculatorTool:
        name = "calculator"
        description = "calc"
        args_schema = None

        async def ainvoke(self, args):
            return "42"

    async def fake_news(expression, cache=None):
        return "新闻分析"

    monkeypatch.setattr(nodes_module, "tools", [FakeCalculatorTool()])
    monkeypatch.setattr(nodes_module, "_internal_call_news_analyst", fake_news)

    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"},
            {"name": "news_analyst", "args": {}, "id": "c2"},
        ],
    )
    state = _internal_state_with_sub_agent_call("news_analyst", "c2")
    state["messages"][1] = ai

    result = await nodes_module.call_tool_node(state)

    # 两条 tool_call 各有 ToolMessage；子代理增量写回 state
    assert [m.tool_call_id for m in result["messages"]] == ["c1", "c2"]
    assert result["messages"][0].content == "42"
    assert result["messages"][1].content == "新闻分析"
    assert result["news_report"] == "新闻分析"


@pytest.mark.anyio
async def test_tool_node_runs_multiple_tools_concurrently(monkeypatch):
    import asyncio
    import time

    from langchain_core.messages import AIMessage

    from finabot.agents import nodes as nodes_module

    SLEEP = 0.2

    class SlowTool:
        def __init__(self, name):
            self.name = name
            self.description = "x"
            self.args_schema = None

        async def ainvoke(self, args):
            await asyncio.sleep(SLEEP)
            return f"ok:{self.name}"

    monkeypatch.setattr(nodes_module, "tools", [SlowTool("slow_a"), SlowTool("slow_b"), SlowTool("slow_c")])
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "slow_a", "args": {}, "id": "a1"},
            {"name": "slow_b", "args": {}, "id": "b1"},
            {"name": "slow_c", "args": {}, "id": "c1"},
        ],
    )

    start = time.perf_counter()
    result = await nodes_module.call_tool_node({"messages": [ai], "akshare_cache": {}})
    elapsed = time.perf_counter() - start

    # 3 个工具各 sleep 0.2s，串行约 0.6s；并发应在 ~0.4s 内完成
    assert elapsed < SLEEP * 3 * 0.6, f"tools ran sequentially, elapsed={elapsed:.2f}s"
    # 结果保持原 call 顺序
    assert [m.tool_call_id for m in result["messages"]] == ["a1", "b1", "c1"]


@pytest.mark.anyio
async def test_tool_node_isolates_single_tool_failure(monkeypatch):
    from langchain_core.messages import AIMessage

    from finabot.agents import nodes as nodes_module

    class OkTool:
        name = "ok"
        description = "x"
        args_schema = None

        async def ainvoke(self, args):
            return "ok"

    class BoomTool:
        name = "boom"
        description = "x"
        args_schema = None

        async def ainvoke(self, args):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(nodes_module, "tools", [OkTool(), BoomTool()])
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "ok", "args": {}, "id": "o1"},
            {"name": "boom", "args": {}, "id": "x1"},
        ],
    )

    # 单个工具抛错不再让整轮崩溃，其它工具照常返回
    result = await nodes_module.call_tool_node({"messages": [ai], "akshare_cache": {}})

    by_id = {m.tool_call_id: m.content for m in result["messages"]}
    assert by_id["o1"] == "ok"
    assert "kaboom" in by_id["x1"]


@pytest.mark.anyio
async def test_llm_metrics_capture_usage_and_latency(monkeypatch):
    import finabot.agents.llm as llm_module
    from finabot.agents.telemetry import LLM_METRICS

    class FakeCompletion:
        usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        )
        choices = [SimpleNamespace(message=SimpleNamespace(content="ok"))]

    async def fake_acompletion(**kwargs):
        return FakeCompletion()

    settings = llm_module.LLMSettings(
        provider="zai",
        model="glm-test",
        litellm_model="zai/glm-test",
        api_key="secret",
        timeout_seconds=1,
    )
    LLM_METRICS.reset()
    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)

    await llm_module._internal_acompletion(
        settings,
        [{"role": "user", "content": "hi"}],
        None,
        retry=False,
    )
    snapshot = LLM_METRICS.snapshot()

    assert snapshot["calls"] == 1
    assert snapshot["failures"] == 0
    assert snapshot["prompt_tokens"] == 11
    assert snapshot["completion_tokens"] == 7
    assert snapshot["total_tokens"] == 18
    assert snapshot["recent"][0]["model"] == "zai/glm-test"
    assert snapshot["recent"][0]["latency_ms"] >= 0


class _TransientError(Exception):
    status_code = 429


class _PermanentError(Exception):
    status_code = 400


@pytest.mark.anyio
async def test_internal_acompletion_retries_transient_then_succeeds(monkeypatch):
    import finabot.agents.llm as llm_module

    monkeypatch.setenv("FINABOT_LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("FINABOT_LLM_RETRY_BASE_SECONDS", "0")

    attempts = {"n": 0}

    async def fake_acompletion(**kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise _TransientError("rate limited")
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content="recovered"))],
        )

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    settings = llm_module.LLMSettings(
        provider="openai",
        model="deepseek-v4-pro",
        litellm_model="openai/deepseek-v4-pro",
        api_key="secret",
        timeout_seconds=1,
    )

    response = await llm_module._internal_acompletion(
        settings, [{"role": "user", "content": "hi"}], None, retry=False
    )

    assert attempts["n"] == 3
    assert response.choices[0].message.content == "recovered"


@pytest.mark.anyio
async def test_internal_acompletion_does_not_retry_permanent_error(monkeypatch):
    import finabot.agents.llm as llm_module

    monkeypatch.setenv("FINABOT_LLM_MAX_RETRIES", "3")

    attempts = {"n": 0}

    async def fake_acompletion(**kwargs):
        attempts["n"] += 1
        raise _PermanentError("bad request")

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    settings = llm_module.LLMSettings(
        provider="openai",
        model="deepseek-v4-pro",
        litellm_model="openai/deepseek-v4-pro",
        api_key="secret",
        timeout_seconds=1,
    )

    with pytest.raises(_PermanentError):
        await llm_module._internal_acompletion(
            settings, [{"role": "user", "content": "hi"}], None, retry=False
        )

    # 客户端错误（400）不可重试，应只尝试一次后立即抛出。
    assert attempts["n"] == 1

