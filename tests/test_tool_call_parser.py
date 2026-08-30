"""Tests for the multi-tool-call text fallback parser."""

from finabot.agents.nodes import extract_tool_calls_from_content


def test_parses_single_tool_with_one_arg():
    text = "market_analyst<tool_call><arg_key>expression</arg_key><arg_value>贵州茅台</arg_value></tool_call>"
    calls = extract_tool_calls_from_content(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "market_analyst"
    assert calls[0]["args"] == {"expression": "贵州茅台"}


def test_parses_multiple_args():
    text = "hold_analysis_pipeline<tool_call><arg_key>expression</arg_key><arg_value>贵州茅台</arg_value><arg_key>debate_mode</arg_key><arg_value>true</arg_value></tool_call>"
    calls = extract_tool_calls_from_content(text)
    assert len(calls) == 1
    assert calls[0]["args"] == {"expression": "贵州茅台", "debate_mode": "true"}


def test_parses_multiple_tool_blocks():
    text = (
        "market_analyst<tool_call><arg_key>expression</arg_key><arg_value>茅台</arg_value></tool_call>"
        "news_analyst<tool_call><arg_key>expression</arg_key><arg_value>平安</arg_value></tool_call>"
    )
    calls = extract_tool_calls_from_content(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "market_analyst"
    assert calls[1]["name"] == "news_analyst"


def test_parses_name_inside_block_before_args():
    text = "<tool_call>market_analyst<arg_key>expression</arg_key><arg_value>茅台</arg_value></tool_call>"
    calls = extract_tool_calls_from_content(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "market_analyst"


def test_function_name_wrapped_in_function_tag():
    text = "<tool_call><function>news_analyst</function><arg_key>expression</arg_key><arg_value>新闻</arg_value></tool_call>"
    calls = extract_tool_calls_from_content(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "news_analyst"


def test_empty_content_returns_empty():
    assert extract_tool_calls_from_content("") == []


def test_no_tool_markup_returns_empty():
    assert extract_tool_calls_from_content("这是普通文本，没有工具调用") == []