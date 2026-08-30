"""Eval task schema and JSON loader.

The evaluation report requires each task to carry: task_id, suite, as_of,
question, allowed_sources, forbidden_actions, output_schema, reference
claims and calculations, acceptable variants, hard gates, graders, and
budget. Tasks are stored as JSON files under ``eval/tasks/{suite}/``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalTask:
    task_id: str
    suite: str = "dev"  # dev | regression | hidden
    as_of: str = ""
    question: str = ""
    allowed_sources: list[str] = field(default_factory=lambda: ["cninfo", "sse", "eastmoney"])
    forbidden_actions: list[str] = field(default_factory=list)
    output_schema: str = "analyst_report"
    reference_claims: list[str] = field(default_factory=list)
    reference_calculations: list[dict[str, Any]] = field(default_factory=list)
    acceptable_variants: list[str] = field(default_factory=list)
    hard_gates: list[str] = field(default_factory=list)
    graders: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=lambda: {
        "max_llm_calls": 8,
        "max_tokens": 60000,
        "max_cost_cny": 5.0,
        "max_seconds": 300,
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTask":
        default_budget = {
            "max_llm_calls": 8,
            "max_tokens": 60000,
            "max_cost_cny": 5.0,
            "max_seconds": 300,
        }
        default_sources = ["cninfo", "sse", "eastmoney"]
        budget = dict(data.get("budget") or {})
        merged_budget = {**default_budget, **budget}
        allowed = list(data.get("allowed_sources")) if data.get("allowed_sources") else default_sources
        return cls(
            task_id=str(data.get("task_id", "")),
            suite=str(data.get("suite", "dev")),
            as_of=str(data.get("as_of", "")),
            question=str(data.get("question", "")),
            allowed_sources=allowed,
            forbidden_actions=list(data.get("forbidden_actions") or []),
            output_schema=str(data.get("output_schema", "analyst_report")),
            reference_claims=list(data.get("reference_claims") or []),
            reference_calculations=list(data.get("reference_calculations") or []),
            acceptable_variants=list(data.get("acceptable_variants") or []),
            hard_gates=list(data.get("hard_gates") or []),
            graders=list(data.get("graders") or []),
            budget=merged_budget,
        )


def load_task(path: str | os.PathLike[str]) -> EvalTask:
    """Load a single task from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalTask.from_dict(data)


def load_tasks(suite_dir: str | os.PathLike[str]) -> list[EvalTask]:
    """Load all `.json` task files from a directory, sorted by task_id."""
    root = Path(suite_dir)
    if not root.is_dir():
        return []
    tasks: list[EvalTask] = []
    for path in sorted(root.glob("*.json")):
        tasks.append(load_task(path))
    tasks.sort(key=lambda t: t.task_id)
    return tasks


def find_task_root() -> Path:
    """Return the project-level ``eval/tasks`` directory."""
    # Start from finabot/eval/ (this module) and go up two levels
    here = Path(__file__).resolve().parent  # finabot/eval/
    project_root = here.parent.parent  # repo root
    candidate = project_root / "eval" / "tasks"
    if candidate.is_dir():
        return candidate
    # Try cwd-based fallback
    fallback = Path(os.getcwd()) / "eval" / "tasks"
    if fallback.is_dir():
        return fallback
    return candidate


def load_task_by_id(task_id: str) -> EvalTask | None:
    """Load a task by its id from any suite directory.

    Files are named e.g. ``t001_timing_leak.json`` while the embedded
    ``task_id`` is ``t001``, so we load candidates and compare the field.
    """
    root = find_task_root()
    for suite_dir in ["dev", "regression", "hidden"]:
        suite_path = root / suite_dir
        if not suite_path.is_dir():
            continue
        for path in sorted(suite_path.glob(f"{task_id}*.json")):
            task = load_task(path)
            if task.task_id == task_id:
                return task
    return None