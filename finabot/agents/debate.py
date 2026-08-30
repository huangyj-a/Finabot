"""Shared debate-context helpers for bull/bear researcher routes."""

from __future__ import annotations

from typing import Any


def _internal_get_debate_context(state: dict[str, Any]) -> dict[str, Any]:
    debate_context = dict(state.get("debate_context", {}) or {})
    debate_context.setdefault("history", "")
    debate_context.setdefault("bull_history", "")
    debate_context.setdefault("bear_history", "")
    debate_context.setdefault("current_response", "")
    debate_context.setdefault("last_bull_argument", "")
    debate_context.setdefault("last_bear_argument", "")
    debate_context.setdefault("last_speaker", None)
    debate_context.setdefault("count", 0)
    return debate_context


def _internal_record_debate_argument(debate_context: dict[str, Any], speaker: str, content: str) -> dict[str, Any]:
    updated = dict(debate_context)
    label = "Bull Analyst" if speaker == "bull" else "Bear Analyst"
    argument = f"{label}: {content}"
    history = updated.get("history", "")
    speaker_history_key = "bull_history" if speaker == "bull" else "bear_history"
    last_argument_key = "last_bull_argument" if speaker == "bull" else "last_bear_argument"

    updated["history"] = f"{history}\n{argument}".strip()
    updated[speaker_history_key] = f"{updated.get(speaker_history_key, '')}\n{argument}".strip()
    updated[last_argument_key] = argument
    updated["current_response"] = argument
    updated["last_speaker"] = speaker
    updated["count"] = int(updated.get("count", 0) or 0) + 1
    updated["in_progress"] = None
    return updated
