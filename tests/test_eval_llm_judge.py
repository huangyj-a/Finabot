"""Tests for isolated LLM judges (news/bear/synthesis dimensions)."""

import asyncio
from types import SimpleNamespace

from finabot.eval import llm_judge
from finabot.eval.llm_judge import (
    _internal_parse_score,
    judge_dimension,
    judge_quality_dimensions,
)


def test_parse_score_from_json():
    assert _internal_parse_score('{"score": 0.8, "reason": "机制链完整"}') == 0.8


def test_parse_score_from_plain_text():
    assert _internal_parse_score("评分理由…… score: 0.6") == 0.6


def test_parse_score_clamps_to_range():
    assert _internal_parse_score('{"score": 1.5}') == 1.0
    assert _internal_parse_score('{"score": -0.2}') == 0.0


def test_parse_score_returns_none_on_garbage():
    assert _internal_parse_score("完全没有分数") is None


def test_judge_dimension_parses_json(monkeypatch):
    async def fake_llm(messages=None, system_prompt=None, stream_label=None, **kwargs):
        return SimpleNamespace(content='{"score": 0.7, "reason": "新闻机制链较完整"}')

    monkeypatch.setattr(llm_judge, "litellm_glm_call", fake_llm)
    result = asyncio.run(judge_dimension("news_reasoning", "问题", "答案", {}))
    assert result["dimension"] == "news_reasoning"
    assert result["score"] == 0.7


def test_judge_dimension_error_falls_back_to_none(monkeypatch):
    async def fake_llm(messages=None, system_prompt=None, stream_label=None, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(llm_judge, "litellm_glm_call", fake_llm)
    result = asyncio.run(judge_dimension("bear_counter", "问题", "答案", {}))
    assert result["score"] is None
    assert "judge_error" in result["reason"]


def test_judge_dimension_unknown_dimension():
    result = asyncio.run(judge_dimension("nope", "问题", "答案", {}))
    assert result["score"] is None


def test_judge_quality_dimensions_returns_only_successful(monkeypatch):
    responses = {
        "news_reasoning": 0.8,
        "bear_counter": 0.6,
        "agent_synthesis": 0.9,
    }

    async def fake_llm(messages=None, system_prompt=None, stream_label=None, **kwargs):
        # 通过 system_prompt 内容反查维度
        text = system_prompt or ""
        if "新闻推理" in text:
            score = responses["news_reasoning"]
        elif "看空反证" in text:
            score = responses["bear_counter"]
        elif "多 Agent 综合" in text:
            score = responses["agent_synthesis"]
        else:
            raise RuntimeError("unknown judge")
        return SimpleNamespace(content=f'{{"score": {score}, "reason": "ok"}}')

    monkeypatch.setattr(llm_judge, "litellm_glm_call", fake_llm)
    scores = asyncio.run(judge_quality_dimensions("问题", "答案", {"news": "新闻报告"}))
    assert scores == responses


def test_judge_quality_dimensions_skips_failed_dimension(monkeypatch):
    async def fake_llm(messages=None, system_prompt=None, stream_label=None, **kwargs):
        text = system_prompt or ""
        if "新闻推理" in text:
            raise RuntimeError("down")
        return SimpleNamespace(content='{"score": 0.5, "reason": "ok"}')

    monkeypatch.setattr(llm_judge, "litellm_glm_call", fake_llm)
    scores = asyncio.run(judge_quality_dimensions("问题", "答案", {}))
    # news 失败被跳过，bear/synthesis 成功
    assert "news_reasoning" not in scores
    assert scores["bear_counter"] == 0.5
    assert scores["agent_synthesis"] == 0.5


def test_harness_merges_judge_scores(tmp_path, monkeypatch):
    import finabot.eval.harness as harness_module
    from finabot.eval.harness import EvalRunner
    from finabot.eval.tasks import load_task_by_id

    async def _canned_run_one(task, ctx):
        text = "结论：适合持有。新闻显示订单增长但监管风险上升，多空观点存在冲突，触发条件为提价预期。"
        trace = {"messages": [], "reports": {"news": "新闻", "bull": "看涨", "bear": "看跌"}}
        return text, {"latency_ms": 100.0, "trace": trace}

    async def fake_judge(question, final_text, reports=None):
        return {"news_reasoning": 0.9, "bear_counter": 0.8, "agent_synthesis": 0.7}

    monkeypatch.setattr(harness_module, "judge_quality_dimensions", fake_judge)

    task = load_task_by_id("t001")
    runner = EvalRunner(reports_root=tmp_path, run_one=_canned_run_one, enable_llm_judge=True)
    records = asyncio.run(runner.run_task(task, trials=1))

    record = records[0]
    assert record.judge_scores == {"news_reasoning": 0.9, "bear_counter": 0.8, "agent_synthesis": 0.7}
    # 三个 LLM 维度用 judge 分，体现在质量明细的 raw 上
    assert record.quality_detail["details"]["news_reasoning"]["raw"] == 0.9
    assert record.quality_detail["details"]["bear_counter"]["raw"] == 0.8
    assert record.quality_detail["details"]["agent_synthesis"]["raw"] == 0.7