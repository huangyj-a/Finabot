"""Tests for the eval trial runner (offline, canned run_one)."""

import asyncio
import json
from pathlib import Path

from finabot.eval.harness import EvalRunner
from finabot.eval.tasks import load_task_by_id


async def _canned_run_one(task, ctx):
    """Deterministic fake executor: returns a well-formed answer.

    Used to test the grading/trace pipeline without calling the LLM.
    """
    text = (
        "结论：贵州茅台适合持有。核心判断：2026-05-29 收盘价 1792.4 元，"
        "20日涨跌幅 2.35%，PE 历史分位约 40%。看多逻辑：品牌壁垒、现金流强劲。"
        "看空风险：估值不低、批价波动。情景：乐观触发提价预期，悲观触发业绩不及预期。"
        "风险提示：本回答不构成投资建议。来源：东方财富 2026-05-29。"
    )
    trace = {
        "messages": [
            {"type": "ai", "content": "调用 stock_a_history"},
            {"type": "tool", "content": "{\"tool\": \"stock_a_history\", \"rows\": 22}"},
            {"type": "ai", "content": text},
        ],
        "evidence_registry": {"stock_a_history@0": {"tool": "stock_a_history", "data_as_of": "2026-05-29"}},
        "run_meta": {"llm_calls": 3},
        "reports": {"market": "", "news": "", "bull": "", "bear": "", "fundamentals": ""},
    }
    return text, {"latency_ms": 1234.5, "trace": trace}


def test_runner_grades_and_writes_report(tmp_path, monkeypatch):
    task = load_task_by_id("t001")
    assert task is not None

    runner = EvalRunner(reports_root=tmp_path, run_one=_canned_run_one, quality_threshold=75.0)
    records = asyncio.run(runner.run_task(task, trials=2))

    assert len(records) == 2
    for record in records:
        assert record.pass_gates is True, record.failed_gates
        assert record.quality > 0
        assert record.severe is False
        assert record.latency_ms == 1234.5
        assert record.calc["pass_ratio"] == 1.0

    # 报告落盘
    report_dirs = list(tmp_path.iterdir())
    assert report_dirs, "report dir should be written"
    payload = json.loads((report_dirs[0] / "t001.json").read_text(encoding="utf-8"))
    assert payload["run_id"].startswith("t001-")
    assert payload["task_id"] == "t001"
    assert len(payload["trials"]) == 2


def test_runner_catches_future_leak(tmp_path, monkeypatch):
    async def _leaky_run_one(task, ctx):
        # 用 task.as_of 之后一年的日期，确保"未来泄漏"命中
        from datetime import datetime, timedelta
        future = (datetime.fromisoformat(task.as_of) + timedelta(days=365)).strftime("%Y-%m-%d")
        text = f"{future} 的公告（as_of 之后）显示利好，建议关注。"
        return text, {"latency_ms": 10.0, "trace": {"messages": []}}

    task = load_task_by_id("t001")
    runner = EvalRunner(reports_root=tmp_path, run_one=_leaky_run_one)
    records = asyncio.run(runner.run_task(task, trials=1))
    assert records[0].future_leak is True
    assert records[0].severe is True
    assert "no_future_leak" in records[0].failed_gates


def test_runner_marks_refusal_task(tmp_path, monkeypatch):
    async def _refusal_run_one(task, ctx):
        return "无法提供具体买卖建议，以下是风险教育与研究方法：……", {
            "latency_ms": 5.0, "trace": {"messages": []},
        }

    task = load_task_by_id("t007")
    runner = EvalRunner(reports_root=tmp_path, run_one=_refusal_run_one)
    records = asyncio.run(runner.run_task(task, trials=1))
    assert records[0].refusal_expected is True
    assert records[0].refusal_given is True