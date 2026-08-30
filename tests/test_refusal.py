"""Tests for the compliance refusal classifier."""

from finabot.agents.refusal import classify_question, maybe_append_refusal_note


def test_safe_research_question():
    decision = classify_question("请分析贵州茅台的估值与基本面")
    assert decision.level == "safe"


def test_specific_buy_advice_is_refused():
    decision = classify_question("我有50万，帮我买入中国平安，应该买多少仓位")
    assert decision.level == "refuse_specific_advice"


def test_guarantee_terms_are_refused():
    decision = classify_question("这个产品稳赚不赔，年化30%，可以买吗")
    assert decision.level == "refuse_specific_advice"


def test_research_qualified_mention_is_caution():
    decision = classify_question("请讲解买入和卖出的交易规则与费用区别")
    assert decision.level == "caution"


def test_maybe_append_refusal_note_safe_returns_unchanged():
    prompt = "系统提示"
    assert maybe_append_refusal_note("请分析估值", prompt) == prompt


def test_maybe_append_refusal_note_refusal_appends():
    prompt = "系统提示"
    result = maybe_append_refusal_note("帮我买入中国平安", prompt)
    assert "合规边界" in result
    assert result.startswith(prompt)