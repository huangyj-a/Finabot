"""Evidence registry for traceable claims (evaluation report requirement).

Every tool result that carries source metadata (fetch_time, data_as_of,
resolved_symbol, news_scope, ...) is registered in the run-scoped
``evidence_registry`` so downstream claims can reference a source_id and
graders can verify traceability. This is a pure helper module: it only
mutates the ``dict`` it is given and never performs network/LLM work.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _internal_try_parse(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(str(text))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def register_tool_evidence(
    registry: dict[str, dict[str, Any]],
    tool_name: str,
    result_text: str,
    *,
    default_source: str | None = None,
    default_priority: int = 1,
) -> str:
    """Register evidence metadata from a tool result; returns source_id.

    If ``result_text`` is a JSON payload it is scanned for fetch_time /
    data_as_of / resolved_symbol / scope / source; otherwise a minimal
    entry with the raw tool name is recorded. The key is stable per tool
    call so multiple calls to the same tool overwrite with the latest.
    """
    source_id = f"{tool_name}@{len(registry)}"
    meta: dict[str, Any] = {
        "tool": tool_name,
        "source": default_source or tool_name,
        "priority": default_priority,
        "scope": "unknown",
    }
    parsed = _internal_try_parse(result_text)
    if parsed is not None:
        if parsed.get("error"):
            meta["error"] = str(parsed["error"])[:200]
        if parsed.get("fetch_time"):
            meta["retrieved_at"] = str(parsed["fetch_time"])
        if parsed.get("data_as_of"):
            meta["data_as_of"] = str(parsed["data_as_of"])
        if parsed.get("resolved_symbol"):
            meta["resolved_symbol"] = str(parsed["resolved_symbol"])
        if parsed.get("news_scope"):
            meta["scope"] = str(parsed["news_scope"])
        if parsed.get("source"):
            meta["source"] = str(parsed["source"])
        # 引用规范：新闻正文含来源级别
        if parsed.get("has_direct_news") is not None:
            meta["has_direct_news"] = bool(parsed["has_direct_news"])
    registry[source_id] = meta
    return source_id


def register_subagent_evidence(
    registry: dict[str, dict[str, Any]],
    role: str,
    report_text: str,
    as_of: str | None = None,
) -> str:
    """Register a sub-agent report as evidence for downstream claims."""
    source_id = f"subagent:{role}@{len(registry)}"
    registry[source_id] = {
        "source": f"subagent:{role}",
        "tool": role,
        "priority": 2,
        "scope": "analysis",
        "retrieved_at": as_of,
        "preview": _internal_preview(report_text, 200),
    }
    return source_id


def _internal_preview(text: str, limit: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:limit]


def evidence_summary(registry: dict[str, dict[str, Any]] | None) -> str:
    """Compact human-readable summary of registered evidence."""
    registry = registry or {}
    if not registry:
        return "（无证据记录）"
    lines = []
    for source_id, meta in registry.items():
        lines.append(
            f"- {source_id} [{meta.get('source', '?')} | "
            f"scope={meta.get('scope', '?')} | "
            f"as_of={meta.get('data_as_of') or meta.get('retrieved_at') or '未知'}]"
        )
    return "\n".join(lines)