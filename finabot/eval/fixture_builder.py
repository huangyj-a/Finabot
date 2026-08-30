"""Frozen fixture snapshot builder (评估报告: 冻结数据离线套件).

Provides the JSON-safe conversion and snapshot assembly used by
``scripts/gen_fixtures.py``. Pure functions here are offline-testable; the
actual AKShare sampling lives in the script (network-bound).
"""

from __future__ import annotations

import json
from typing import Any


def records_from_frame(df, limit: int = 60) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to a JSON-safe list of records.

    NaN/Inf are serialized as ``null`` by pandas ``to_json``, so the result is
    always valid JSON. Returns [] for empty/None frames.
    """
    if df is None or getattr(df, "empty", True):
        return []
    return json.loads(df.head(int(limit)).to_json(orient="records", force_ascii=False))


def assemble_snapshot(meta: dict[str, Any], fetches: dict[str, Any]) -> dict[str, Any]:
    """Assemble a snapshot dict: ``_meta`` + one key per fetcher name.

    ``fetches`` values are lists of records (already JSON-safe) or raw frames;
    raw frames are converted via ``records_from_frame``.
    """
    snapshot: dict[str, Any] = {"_meta": dict(meta)}
    for name, value in fetches.items():
        if value is None:
            snapshot[name] = []
        elif isinstance(value, list):
            snapshot[name] = value
        elif hasattr(value, "to_json"):
            snapshot[name] = records_from_frame(value)
        else:
            snapshot[name] = str(value)
    return snapshot


def write_snapshot(path, snapshot: dict[str, Any]) -> None:
    """Write a snapshot dict to ``path`` as UTF-8 JSON."""
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")