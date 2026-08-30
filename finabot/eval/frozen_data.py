"""Offline frozen-data layer for evaluation (评估报告: 冻结数据离线套件).

Each task runs against a frozen snapshot instead of live AKShare so trials
are reproducible and future-information leaks are impossible. The snapshot
is served by intercepting the same seams the unit tests use
(``monkeypatch.setattr(aktools.ak, ...)``), packaged as a reusable factory.

Two modes:
- frozen: replace AKShare fetchers with snapshot data (no network).
- shadow: do not intercept; only record metadata for drift detection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from finabot.eval.tasks import EvalTask


class FrozenData:
    """Load and serve a per-task frozen data snapshot."""

    def __init__(self, task: EvalTask, fixtures_root: str | os.PathLike[str] | None = None):
        self.task = task
        if fixtures_root is None:
            here = Path(__file__).resolve().parent  # finabot/eval/
            project_root = here.parent.parent
            fixtures_root = os.getenv("FINABOT_EVAL_FIXTURES", str(project_root / "eval" / "fixtures"))
        self.fixtures_root = Path(fixtures_root)
        self.task_dir = self.fixtures_root / task.task_id
        self.snapshot: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        snapshot_path = self.task_dir / "snapshot.json"
        if snapshot_path.is_file():
            self.snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        else:
            self.snapshot = {}

    @property
    def available(self) -> bool:
        return bool(self.snapshot)

    def get_akshare_frame(self, fetcher_name: str) -> Any:
        """Return a pandas DataFrame for a named AKShare fetcher from snapshot.

        Returns an empty DataFrame (with the expected columns) when missing so
        tools degrade gracefully instead of raising.
        """
        import pandas as pd

        records = self.snapshot.get(fetcher_name)
        if not records:
            return pd.DataFrame()
        if isinstance(records, dict) and "rows" in records:
            records = records.get("rows", [])
        return pd.DataFrame(records)

    def get_news_payload(self) -> dict[str, Any] | None:
        """Return the frozen news tool payload (same shape as get_stock_news_unified)."""
        raw = self.snapshot.get("stock_news")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(str(raw))
        except (ValueError, TypeError):
            return None


def install_frozen_akshare(frozen: FrozenData, monkeypatch) -> None:
    """Intercept the AKShare module's fetchers with snapshot data.

    Usage (pytest-style monkeypatch):
        monkeypatch = ...
        install_frozen_akshare(FrozenData(task), monkeypatch)

    Only fetchers present in the snapshot are intercepted; others are left
    untouched so the harness can still exercise the real path in shadow mode.
    """
    import finabot.tools.akshare_tools as aktools

    for name in frozen.snapshot:
        if name in {"stock_news", "_meta"}:
            continue
        fetcher = getattr(aktools.ak, name, None)
        if fetcher is None:
            continue

        def _make_factory(fetcher_name: str) -> Callable[..., Any]:
            def _fake(*args, **kwargs):
                return frozen.get_akshare_frame(fetcher_name)
            return _fake

        monkeypatch.setattr(aktools.ak, name, _make_factory(name))


class patch_akshare:
    """Context manager that patches AKShare fetchers for the process lifetime.

    Used by the CLI runner (no pytest monkeypatch available). Restores the
    original fetchers on exit. Only snapshot fetchers are patched.
    """

    def __init__(self, frozen: FrozenData):
        self.frozen = frozen
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> "patch_akshare":
        import finabot.tools.akshare_tools as aktools

        for name in self.frozen.snapshot:
            if name in {"stock_news", "_meta"}:
                continue
            if not hasattr(aktools.ak, name):
                continue
            self._saved[name] = getattr(aktools.ak, name)
            setattr(aktools.ak, name, self._make_factory(name))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import finabot.tools.akshare_tools as aktools

        for name, original in self._saved.items():
            setattr(aktools.ak, name, original)

    def _make_factory(self, fetcher_name: str) -> Callable[..., Any]:
        def _fake(*args, **kwargs):
            return self.frozen.get_akshare_frame(fetcher_name)
        return _fake


def shadow_mode_enabled() -> bool:
    return os.getenv("FINABOT_EVAL_SHADOW", "0").strip().lower() in {"1", "true", "yes"}