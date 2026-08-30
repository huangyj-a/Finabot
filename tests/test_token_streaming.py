"""Tests for token-level streaming (llm.py stream path + streaming.py sink)."""

import asyncio
from types import SimpleNamespace

from finabot.agents.streaming import reset_token_sink, set_token_sink


def _settings(monkeypatch, llm_module):
    monkeypatch.setattr(
        llm_module,
        "get_llm_settings",
        lambda: llm_module.LLMSettings(
            provider="zai",
            model="glm-test",
            litellm_model="zai/glm-test",
            api_key="k",
            timeout_seconds=5,
        ),
    )


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _stream_gen(items):
    async def gen():
        for item in items:
            yield item
    return gen()


def test_stream_completion_forwards_tokens_and_builds_message(monkeypatch):
    import finabot.agents.llm as llm_module

    captured = []

    async def sink(label, text):
        captured.append((label, text))

    # stream=True 时 litellm 的 acompletion 直接返回异步生成器（不是协程），
    # 因此 fake 必须是同步函数并返回异步生成器。
    def fake_acompletion(**kwargs):
        if kwargs.get("stream"):
            return _stream_gen([_chunk(content="贵州"), _chunk(content="茅台"), _chunk(content="值得持有")])
        return SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))]
        )

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    _settings(monkeypatch, llm_module)

    token = set_token_sink(sink)
    try:
        msg = asyncio.run(llm_module.litellm_glm_call([], stream_label="news_analyst"))
    finally:
        reset_token_sink(token)

    assert msg.content == "贵州茅台值得持有"
    assert captured == [
        ("news_analyst", "贵州"),
        ("news_analyst", "茅台"),
        ("news_analyst", "值得持有"),
    ]
    assert msg.tool_calls == []


def test_stream_completion_reconstructs_tool_calls(monkeypatch):
    import finabot.agents.llm as llm_module

    async def sink(label, text):
        pass

    def fake_acompletion(**kwargs):
        if kwargs.get("stream"):
            return _stream_gen(
                [
                    _chunk(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(name="hold_analysis", arguments='{"express'),
                            )
                        ]
                    ),
                    _chunk(
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                function=SimpleNamespace(name=None, arguments='ion": "茅台"}'),
                            )
                        ]
                    ),
                ]
            )
        return SimpleNamespace(
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))]
        )

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    _settings(monkeypatch, llm_module)

    token = set_token_sink(sink)
    try:
        msg = asyncio.run(llm_module.litellm_glm_call([], stream_label="supervisor"))
    finally:
        reset_token_sink(token)

    assert msg.content == ""
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["function"]["name"] == "hold_analysis"
    assert msg.tool_calls[0]["function"]["arguments"] == '{"expression": "茅台"}'


def test_non_streamable_label_uses_whole_response(monkeypatch):
    import finabot.agents.llm as llm_module

    captured = {}

    async def sink(label, text):
        captured["sink_called"] = True

    async def fake_acompletion(**kwargs):
        captured["stream_flag"] = kwargs.get("stream", False)
        # 非流式路径：acompletion 返回完整 response
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="整段结论", tool_calls=[]))],
        )

    monkeypatch.setattr(llm_module, "acompletion", fake_acompletion)
    _settings(monkeypatch, llm_module)

    token = set_token_sink(sink)
    try:
        msg = asyncio.run(llm_module.litellm_glm_call([], stream_label="bear_researcher"))
    finally:
        reset_token_sink(token)

    assert captured.get("stream_flag") is False
    assert "sink_called" not in captured
    assert msg.content == "整段结论"
