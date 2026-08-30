"""Trial runner: execute a task, capture a trace, grade it, write a report.

The runner is decoupled from the execution engine: a ``run_one`` callable
takes the task (and an optional seed) and returns ``(final_text, trace)``.
A default runner drives the LangGraph directly with frozen data installed,
which keeps the harness offline and reproducible.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from finabot.eval.frozen_data import FrozenData, install_frozen_akshare, patch_akshare
from finabot.eval.graders import (
    check_fact_traceability,
    check_reference_calculations,
    deterministic_dimension_scores,
    run_hard_gates,
    score_quality,
)
from finabot.eval.llm_judge import judge_quality_dimensions
from finabot.eval.tasks import EvalTask

RunOneFn = Callable[[EvalTask, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]


async def _default_run_one(task: EvalTask, ctx: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Run the Finabot graph directly with frozen data.

    ctx may carry: monkeypatch (pytest fixture), frozen (FrozenData).
    When monkeypatch is absent (CLI), the ``patch_akshare`` context manager
    patches fetchers for the process lifetime and restores them after.
    """
    from langchain_core.messages import HumanMessage

    from finabot.agents.core import _RUN_SCOPED_STATE_DEFAULTS
    from finabot.graph.graph import build_graph

    monkeypatch = ctx.get("monkeypatch")
    frozen: FrozenData = ctx["frozen"]
    patcher = None
    if monkeypatch is not None:
        install_frozen_akshare(frozen, monkeypatch)
    elif frozen.available:
        patcher = patch_akshare(frozen)
        patcher.__enter__()

    started = time.perf_counter()
    try:
        single_agent = bool(ctx.get("single_agent", False)) or os.getenv("FINABOT_SINGLE_AGENT", "0").strip().lower() in {"1", "true", "yes", "on"}
        graph = build_graph(checkpointer=None, single_agent=single_agent)
        state: dict[str, Any] = {
            "messages": [HumanMessage(content=task.question)],
            "session_key": f"eval:{task.task_id}",
            "user_id": "eval",
            "memories": [],
            "as_of": task.as_of or None,
            "run_meta": {"started_at": task.as_of, "llm_calls": 0, "subagent_timeouts": []},
        }
        for field_name, factory in _RUN_SCOPED_STATE_DEFAULTS.items():
            if field_name not in state:
                state[field_name] = factory()

        final = await graph.ainvoke(
            state,
            config={"recursion_limit": max(1, int(os.getenv("FINABOT_MAX_RECURSION", "16")))},
        )
        reply = final["messages"][-1].content
    finally:
        if patcher is not None:
            patcher.__exit__(None, None, None)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    trace: dict[str, Any] = {
        "messages": [
            {"type": getattr(m, "type", "?"), "content": str(getattr(m, "content", ""))[:1000]}
            for m in final.get("messages", [])
        ],
        "evidence_registry": final.get("evidence_registry", {}),
        "run_meta": final.get("run_meta", {}),
        "reports": {
            "market": final.get("market_report", ""),
            "news": final.get("news_report", ""),
            "bull": final.get("bull_report", ""),
            "bear": final.get("bear_report", ""),
            "fundamentals": final.get("fundamentals_report", ""),
        },
    }
    return str(reply), {"latency_ms": elapsed_ms, "trace": trace}


@dataclass
class TrialRecord:
    task_id: str
    trial: int
    run_id: str
    final_text: str
    pass_gates: bool
    failed_gates: list[str]
    quality: float
    quality_detail: dict[str, Any]
    severe: bool
    latency_ms: float
    calc: dict[str, Any]
    future_leak: bool
    tool_errors: int
    judge_scores: dict[str, float] = field(default_factory=dict)
    fact_traceability: dict[str, Any] = field(default_factory=dict)
    refusal_expected: bool = False
    refusal_given: bool = False
    refusal_appropriate: bool = True
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "run_id": self.run_id,
            "pass_gates": self.pass_gates,
            "failed_gates": self.failed_gates,
            "quality": self.quality,
            "quality_detail": self.quality_detail,
            "severe": self.severe,
            "latency_ms": self.latency_ms,
            "calc": self.calc,
            "future_leak": self.future_leak,
            "tool_errors": self.tool_errors,
            "judge_scores": self.judge_scores,
            "fact_traceability": self.fact_traceability,
            "refusal_expected": self.refusal_expected,
            "refusal_given": self.refusal_given,
            "refusal_appropriate": self.refusal_appropriate,
            "final_text": self.final_text[:4000],
            "trace": self.trace,
        }


class EvalRunner:
    """Run one task for N trials and collect records."""

    def __init__(
        self,
        *,
        fixtures_root: str | os.PathLike[str] | None = None,
        reports_root: str | os.PathLike[str] | None = None,
        run_one: RunOneFn | None = None,
        quality_threshold: float = 75.0,
        enable_llm_judge: bool = False,
    ):
        self.fixtures_root = fixtures_root
        if reports_root is None:
            here = Path(__file__).resolve().parent
            project_root = here.parent.parent
            reports_root = project_root / "eval" / "reports"
        self.reports_root = Path(reports_root)
        self.run_one = run_one or _default_run_one
        self.quality_threshold = quality_threshold
        self.enable_llm_judge = enable_llm_judge

    async def run_task(
        self,
        task: EvalTask,
        trials: int = 5,
        *,
        monkeypatch=None,
        seed: int | None = None,
    ) -> list[TrialRecord]:
        """Run a task for ``trials`` times, returning graded records."""
        run_id = f"{task.task_id}-{uuid.uuid4().hex[:8]}"
        records: list[TrialRecord] = []
        for trial in range(1, trials + 1):
            frozen = FrozenData(task, self.fixtures_root)
            ctx: dict[str, Any] = {"frozen": frozen, "monkeypatch": monkeypatch}
            final_text, extra = await self.run_one(task, ctx)
            judge_scores: dict[str, float] = {}
            if self.enable_llm_judge:
                reports = (extra.get("trace") or {}).get("reports", {})
                judge_scores = await judge_quality_dimensions(task.question, final_text, reports)
            record = self._grade(task, run_id, trial, final_text, extra, judge_scores)
            records.append(record)
        self._write_report(run_id, task, records)
        return records

    def _grade(
        self,
        task: EvalTask,
        run_id: str,
        trial: int,
        final_text: str,
        extra: dict[str, Any],
        judge_scores: dict[str, float] | None = None,
    ) -> TrialRecord:
        ctx = {"as_of": task.as_of}
        failed_gates = run_hard_gates(final_text, ctx)
        pass_gates = not failed_gates

        calc = check_reference_calculations(final_text, task.reference_calculations)
        dims = deterministic_dimension_scores(final_text)
        judge_scores = judge_scores or {}
        dims.update(judge_scores)  # LLM Judge 覆盖新闻/反证/综合三个维度
        quality = score_quality(final_text, ctx, dimension_scores=dims)
        severe = bool(failed_gates)

        trace = extra.get("trace", {})
        tool_errors = _internal_count_tool_errors(trace)
        future_leak = "no_future_leak" in failed_gates
        evidence_text = _internal_evidence_text(trace)
        fact_traceability = check_fact_traceability(final_text, evidence_text)

        # 拒绝准确性：从问题与门禁推导（可被 harness 覆盖）
        from finabot.agents.refusal import classify_question
        decision = classify_question(task.question)
        refusal_expected = decision.level != "safe"
        refusal_given = any(marker in final_text for marker in ("不构成投资建议", "无法提供", "不能提供", "风险教育", "仅供参考"))
        refusal_appropriate = True

        return TrialRecord(
            task_id=task.task_id,
            trial=trial,
            run_id=run_id,
            final_text=final_text,
            pass_gates=pass_gates,
            failed_gates=failed_gates,
            quality=quality["total"],
            quality_detail=quality,
            severe=severe,
            latency_ms=float(extra.get("latency_ms", 0) or 0),
            calc=calc,
            future_leak=future_leak,
            tool_errors=tool_errors,
            judge_scores=judge_scores,
            fact_traceability=fact_traceability,
            refusal_expected=refusal_expected,
            refusal_given=refusal_given,
            refusal_appropriate=refusal_appropriate,
            trace=trace,
        )

    def _write_report(self, run_id: str, task: EvalTask, records: list[TrialRecord]) -> Path:
        report_dir = self.reports_root / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "task_id": task.task_id,
            "suite": task.suite,
            "as_of": task.as_of,
            "question": task.question,
            "trials": [r.to_dict() for r in records],
        }
        path = report_dir / f"{task.task_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _internal_count_tool_errors(trace: dict[str, Any]) -> int:
    count = 0
    for message in trace.get("messages", []):
        content = str(message.get("content", ""))
        if "执行失败" in content or "未知工具" in content or "无法解析" in content:
            count += 1
    return count


def _internal_evidence_text(trace: dict[str, Any]) -> str:
    """Concatenate handoff reports + evidence registry into one text blob."""
    parts: list[str] = []
    for value in (trace.get("reports") or {}).values():
        if value:
            parts.append(str(value))
    for meta in (trace.get("evidence_registry") or {}).values():
        if isinstance(meta, dict):
            parts.append(str(meta.get("preview", "")))
            parts.append(str(meta.get("data_as_of", "")))
    return "\n".join(parts)