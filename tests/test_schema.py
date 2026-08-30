"""Tests for the structured output schema parsing."""

from finabot.agents.schema import (
    AnalystOutput,
    analyst_output_to_text,
    parse_analyst_output,
    parse_analyst_outputs,
    structured_output_enabled,
)


def test_structured_output_enabled_default():
    assert structured_output_enabled() is True


def test_parse_json_output():
    text = '{"role": "news_analyst", "as_of": "2026-05-29", "claims": [{"text": "公告发布", "source_ids": ["cninfo@0"], "kind": "fact"}], "confidence": "high", "unknowns": [], "risk_flags": []}'
    output = parse_analyst_output("news_analyst", text, default_as_of="2026-05-29")
    assert output.role == "news_analyst"
    assert output.confidence == "high"
    assert output.claims[0].text == "公告发布"
    assert output.claims[0].source_ids == ["cninfo@0"]


def test_parse_json_wrapped_in_markdown():
    text = '```json\n{"role": "bear", "claims": [], "confidence": "medium", "unknowns": ["缺估值"], "risk_flags": ["估值高"]}\n```'
    output = parse_analyst_output("bear", text, default_as_of="2026-05-29")
    assert output.confidence == "medium"
    assert output.risk_flags == ["估值高"]


def test_freeform_fallback_preserves_text_and_low_confidence():
    text = "这是一段自由文本分析，没有任何结构化 JSON。"
    stored, output = parse_analyst_outputs("fundamental_analyst", text, default_as_of="2026-05-29")
    assert output.confidence == "low"
    assert output.role == "fundamental_analyst"
    # 自由文本回退必须原样保留，不能改写正文
    assert stored == text


def test_empty_text_degrades_gracefully():
    output = parse_analyst_output("news_analyst", "", default_as_of="2026-05-29")
    assert output.confidence == "low"
    assert output.unknowns


def test_disabled_mode_returns_freeform():
    import finabot.agents.schema as schema_mod
    original = schema_mod.structured_output_enabled
    schema_mod.structured_output_enabled = lambda: False
    try:
        text = '{"role": "x", "claims": [], "confidence": "high", "unknowns": [], "risk_flags": []}'
        output = parse_analyst_output("x", text)
        assert output.confidence == "low"
        assert output.claims[0].text == text
    finally:
        schema_mod.structured_output_enabled = original


def test_analyst_output_to_text_joins_claims():
    output = AnalystOutput(
        role="r",
        claims=[
            {"text": "第一段", "kind": "fact"},
            {"text": "第二段", "kind": "inference"},
        ],
        confidence="high",
    )
    assert analyst_output_to_text(output) == "第一段\n\n第二段"