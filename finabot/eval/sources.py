"""Data source grading (评估报告: 数据源分级).

Loads ``eval/policy/sources.json`` and maps a source name to a priority level
(P0 原始 / P1 媒体 / P2 线索). Used by the evidence/citation graders to score
citation priority and to flag social-media claims as unverified leads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _sources_path() -> Path:
    here = Path(__file__).resolve().parent  # finabot/eval/
    project_root = here.parent.parent
    candidate = project_root / "eval" / "policy" / "sources.json"
    if candidate.is_file():
        return candidate
    fallback = Path(os.getcwd()) / "eval" / "policy" / "sources.json"
    return fallback


def load_sources() -> dict[str, list[str]]:
    """Return {level: [source names]} from the policy config."""
    try:
        data = json.loads(_sources_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        "p0": list(data.get("p0", [])),
        "p1": list(data.get("p1", [])),
        "p2": list(data.get("p2", [])),
    }


def source_level(source_name: str) -> str | None:
    """Return 'P0' / 'P1' / 'P2' for a source name, or None if unknown."""
    if not source_name:
        return None
    text = str(source_name)
    for level, names in load_sources().items():
        if any(name in text for name in names):
            return level.upper()
    return None