"""Per-run trace recording (评估报告: 保存完整 trace，供周度抽读).

Writes a best-effort JSON trace per message run under
``memory/runtime/traces/`` so the team can sample passed/failed traces weekly
without relying on the eval harness. Failures are swallowed: tracing must never
break the primary flow.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _traces_dir() -> Path:
    return Path(os.getenv("FINABOT_RUNTIME_DIR", "memory/runtime")) / "traces"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "default"


def write_run_trace(session_key: str, final_state: dict[str, Any] | None) -> Path | None:
    """Write a per-run trace JSON; best-effort (failures ignored)."""
    final_state = final_state or {}
    try:
        traces_dir = _traces_dir()
        traces_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        payload: dict[str, Any] = {
            "session_key": session_key,
            "timestamp": timestamp,
            "messages": [
                {"type": getattr(m, "type", "?"), "content": str(getattr(m, "content", ""))[:500]}
                for m in (final_state.get("messages") or [])[-8:]
            ],
            "reports": {
                "market": final_state.get("market_report", ""),
                "news": final_state.get("news_report", ""),
                "bull": final_state.get("bull_report", ""),
                "bear": final_state.get("bear_report", ""),
                "fundamentals": final_state.get("fundamentals_report", ""),
            },
            "evidence_registry": final_state.get("evidence_registry", {}),
            "risk_flags": final_state.get("risk_flags", []),
            "claims": final_state.get("claims", []),
            "run_meta": final_state.get("run_meta", {}),
        }
        path = traces_dir / f"{_safe_name(session_key)}_{timestamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:  # pragma: no cover - 落盘失败绝不阻断主流程
        return None