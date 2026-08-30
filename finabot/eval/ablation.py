"""Six-ablation comparison harness (评估报告: 六组消融).

Runs the same task set under each ablation config and returns per-config
summary metrics (质量分 / 严重失败率 / 冲突保留率 / P95 延迟 / 成本), so the
team can judge whether multi-agent adds quality or only adds failure surface.

Configs:
- single_agent        FINABOT_SINGLE_AGENT=1 (supervisor + tool, no sub-agents)
- no_bear             FINABOT_NO_BEAR=1 (hold pipeline without bear researcher)
- no_structured        FINABOT_STRUCTURED_OUTPUT=0 (free-text handoff)
- full                 default multi-agent
- random_failure       FINABOT_EVAL_FAIL_NODE=<node> (injected sub-agent failure)
- conflicting_evidence fixture-driven contradictory snapshot (content, not env)

The runner restores env vars after each config, so configs never leak across.
"""

from __future__ import annotations

import os
from typing import Any

from finabot.eval.harness import EvalRunner
from finabot.eval.metrics import summarize_trials
from finabot.eval.tasks import EvalTask

# 随机失败默认命中 hold_analysis_pipeline（持有分析最核心的节点）；
# 测试或高级用法可通过环境变量覆盖为任意子代理名。
DEFAULT_FAIL_NODE = "hold_analysis_pipeline"

ABLATION_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "single_agent", "label": "单 Agent", "env": {"FINABOT_SINGLE_AGENT": "1"}},
    {"name": "no_bear", "label": "无看空角色", "env": {"FINABOT_NO_BEAR": "1"}},
    {"name": "no_structured", "label": "无结构化交接", "env": {"FINABOT_STRUCTURED_OUTPUT": "0"}},
    {"name": "full", "label": "完整多 Agent", "env": {}},
    {"name": "random_failure", "label": "子 Agent 失败", "env": {"FINABOT_EVAL_FAIL_NODE": DEFAULT_FAIL_NODE}},
    {
        "name": "conflicting_evidence",
        "label": "上游证据冲突",
        "env": {},
        "note": "依赖任务专属矛盾快照（fixtures/<task_id>/conflicting/），无则等同 full",
    },
)


async def run_ablations(
    task: EvalTask,
    *,
    trials: int = 1,
    run_one=None,
    fixtures_root: str | os.PathLike[str] | None = None,
    reports_root: str | os.PathLike[str] | None = None,
    quality_threshold: float = 75.0,
    enable_llm_judge: bool = False,
    specs: tuple[dict[str, Any], ...] = ABLATION_SPECS,
) -> dict[str, dict[str, Any]]:
    """Run ``task`` under each ablation config, returning per-config metrics.

    Each result is a ``summarize_trials`` summary augmented with ``label``.
    Env vars are restored after each config so configs never leak across.
    """
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        saved: dict[str, str | None] = {}
        for key, value in spec.get("env", {}).items():
            saved[key] = os.environ.get(key)
            os.environ[key] = str(value)
        try:
            runner = EvalRunner(
                fixtures_root=fixtures_root,
                reports_root=reports_root,
                run_one=run_one,
                quality_threshold=quality_threshold,
                enable_llm_judge=enable_llm_judge,
            )
            records = await runner.run_task(task, trials=trials)
            summary = summarize_trials([r.to_dict() for r in records], quality_threshold)
            summary["label"] = spec["label"]
            results[spec["name"]] = summary
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return results


def compare_ablations(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Condense per-config summaries into a side-by-side comparison."""
    comparison: dict[str, Any] = {}
    for name, summary in results.items():
        comparison[name] = {
            "label": summary.get("label", name),
            "n": summary.get("n"),
            "pass1": summary.get("Pass@1"),
            "severe_rate": summary.get("severe_failure_rate"),
            "latency_p95": (summary.get("latency_ms") or {}).get("p95"),
            "avg_cost_cny": summary.get("avg_cost_cny"),
        }
    return comparison