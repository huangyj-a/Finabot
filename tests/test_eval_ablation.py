"""Tests for the fact gate and six-ablation harness."""

import asyncio

from finabot.eval.ablation import ABLATION_SPECS, compare_ablations, run_ablations
from finabot.eval.graders import check_fact_traceability
from finabot.eval.tasks import load_task_by_id


def test_fact_traceability_all_numbers_in_evidence():
    text = "贵州茅台最新收盘价 1792.4 元，PE 历史分位约 40%，20日涨跌幅 2.35%。"
    evidence = "收盘价 1792.4 元；PE 分位 40%；涨跌幅 2.35%"
    result = check_fact_traceability(text, evidence)
    assert result["pass"] is True
    assert result["untraceable"] == 0
    assert result["ratio"] == 1.0


def test_fact_traceability_flags_untraceable():
    text = "该股目标价 8888 元，市值 9999 亿。"
    evidence = "市值 5000 亿"
    result = check_fact_traceability(text, evidence)
    assert result["pass"] is False
    assert "8888" in result["untraceable_samples"]
    assert "9999" in result["untraceable_samples"]


def test_fact_traceability_missing_marked_passes():
    text = "目标价数据缺失，无法给出具体数值。"
    result = check_fact_traceability(text, "")
    assert result["pass"] is True
    assert result["missing_marked"] is True


def test_fact_traceability_ignores_years_and_small_ints():
    text = "2026 年，3 个交易日，涨幅 15%。"
    result = check_fact_traceability(text, "15%")
    # 年份 2026 与小整数 3 不计入事实数字
    assert result["untraceable"] == 0
    assert result["total"] == 1


async def _canned_run_one(task, ctx):
    text = "结论：适合持有。收盘价 1792.4 元，PE 分位 40%，涨幅 2.35%。"
    trace = {"messages": [], "reports": {"news": "1792.4 元", "fundamentals": "PE 分位 40%"}}
    return text, {"latency_ms": 100.0, "trace": trace}


def test_run_ablations_covers_six_configs(tmp_path):
    task = load_task_by_id("t001")
    assert task is not None
    results = asyncio.run(
        run_ablations(task, trials=1, run_one=_canned_run_one, reports_root=tmp_path)
    )
    assert set(results.keys()) == {spec["name"] for spec in ABLATION_SPECS}
    assert len(results) == 6
    # 每个 config 都产出指标
    for name, summary in results.items():
        assert "Pass@1" in summary
        assert "label" in summary


def test_run_ablations_restores_env(monkeypatch, tmp_path):
    task = load_task_by_id("t001")
    monkeypatch.delenv("FINABOT_SINGLE_AGENT", raising=False)
    monkeypatch.delenv("FINABOT_NO_BEAR", raising=False)
    monkeypatch.delenv("FINABOT_EVAL_FAIL_NODE", raising=False)

    asyncio.run(run_ablations(task, trials=1, run_one=_canned_run_one, reports_root=tmp_path))

    # 环境变量应被恢复（不泄漏）
    import os
    assert os.environ.get("FINABOT_SINGLE_AGENT") is None
    assert os.environ.get("FINABOT_NO_BEAR") is None
    assert os.environ.get("FINABOT_EVAL_FAIL_NODE") is None


def test_compare_ablations_condenses():
    results = {
        "full": {"label": "完整多 Agent", "n": 1, "Pass@1": 0.8, "severe_failure_rate": 0.0, "latency_ms": {"p95": 100.0}, "avg_cost_cny": 0.5},
        "single_agent": {"label": "单 Agent", "n": 1, "Pass@1": 0.7, "severe_failure_rate": 0.0, "latency_ms": {"p95": 80.0}, "avg_cost_cny": 0.3},
    }
    comparison = compare_ablations(results)
    assert comparison["full"]["pass1"] == 0.8
    assert comparison["single_agent"]["latency_p95"] == 80.0
    assert comparison["full"]["label"] == "完整多 Agent"