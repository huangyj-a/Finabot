"""Tests for the structured output schema parsing."""

from finabot.agents.schema import (
    AnalystOutput,
    analyst_output_to_text,
    collect_structured_state,
    parse_analyst_output,
    parse_analyst_outputs,
    parse_subagent_result,
    structured_output_enabled,
    structured_state_update,
)


def test_structured_output_enabled_default_off():
    # 生产默认关闭，评估时设 FINABOT_STRUCTURED_OUTPUT=1 显式开启
    assert structured_output_enabled() is False


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


def test_disabled_mode_returns_freeform(monkeypatch):
    monkeypatch.delenv("FINABOT_STRUCTURED_OUTPUT", raising=False)
    text = '{"role": "x", "claims": [{"text": "主张", "kind": "fact"}], "confidence": "high", "unknowns": [], "risk_flags": []}'
    # 开关关闭时，交接层不做 JSON 拆分，返回原样文本 + 空结构化增量
    display, claims, risk_flags = collect_structured_state("x", text)
    assert display == text
    assert claims == []
    assert risk_flags == []


def test_pure_parser_parses_json_regardless_of_toggle(monkeypatch):
    # 纯解析器 parse_analyst_output 不检查开关，始终尝试解析 JSON
    monkeypatch.delenv("FINABOT_STRUCTURED_OUTPUT", raising=False)
    text = '{"role": "news_analyst", "claims": [{"text": "公告发布", "kind": "fact"}], "confidence": "high", "unknowns": [], "risk_flags": []}'
    output = parse_analyst_output("news_analyst", text)
    assert output.confidence == "high"


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


def test_parse_subagent_result_strips_json_when_enabled(monkeypatch):
    import finabot.agents.schema as schema_mod
    monkeypatch.setenv("FINABOT_STRUCTURED_OUTPUT", "1")
    text = '分析正文：该股基本面稳健。\n\n{"role": "news_analyst", "claims": [{"text": "公告发布", "kind": "fact"}], "confidence": "high", "unknowns": [], "risk_flags": []}'
    display, output = parse_subagent_result("news_analyst", text, "2026-05-29")
    # 正文保留，JSON 已移除
    assert "分析正文：该股基本面稳健" in display
    assert '{"role"' not in display
    assert output.confidence == "high"


def test_collect_structured_state_off_returns_empty(monkeypatch):
    monkeypatch.delenv("FINABOT_STRUCTURED_OUTPUT", raising=False)
    display, claims, risk_flags = collect_structured_state("news_analyst", "自由文本", "2026-05-29")
    assert display == "自由文本"
    assert claims == []
    assert risk_flags == []


def test_structured_state_update_merges_claims(monkeypatch):
    import finabot.agents.schema as schema_mod
    monkeypatch.setenv("FINABOT_STRUCTURED_OUTPUT", "1")
    text = '正文\n\n{"role": "news_analyst", "claims": [{"text": "主张A", "kind": "fact"}], "confidence": "high", "unknowns": [], "risk_flags": ["风险1"]}'
    state = {"claims": [{"text": "已有主张", "kind": "fact"}], "risk_flags": []}
    display, update = structured_state_update("news_analyst", text, state, "2026-05-29")
    assert display == "正文"
    assert len(update["claims"]) == 2  # 已有 + 新增
    assert update["risk_flags"] == ["风险1"]