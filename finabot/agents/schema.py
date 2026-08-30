"""Structured output schema for Finabot sub-agents.

The evaluation report requires every sub-agent to emit claims, evidence,
as_of, confidence, unknowns, and risk_flags. This module defines the
Pydantic models and a tolerant parser: sub-agents are asked to emit JSON
matching ``AnalystOutput``; when the model returns free text instead, the
parser wraps it with ``confidence=low`` rather than failing the round.
The structured layer can be disabled with ``FINABOT_STRUCTURED_OUTPUT=0``
(used by the "no structured handoff" ablation).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - pydantic is an optional runtime dep
    # Fallback minimal types so the module imports even without pydantic.
    BaseModel = object  # type: ignore[assignment,misc]
    Field = lambda default=..., **kwargs: default  # type: ignore[assignment]


class Claim(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)
    as_of: str | None = None
    kind: Literal["fact", "calculation", "inference", "opinion"] = "fact"


class AnalystOutput(BaseModel):
    role: str
    as_of: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    unknowns: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


def structured_output_enabled() -> bool:
    value = os.getenv("FINABOT_STRUCTURED_OUTPUT", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _extract_json_object(text: str) -> str | None:
    """Pull the first balanced JSON object from model text, if any."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_analyst_output(role: str, text: str, default_as_of: str | None = None) -> AnalystOutput:
    """Parse a sub-agent response into ``AnalystOutput``.

    If the model returned structured JSON, it is validated (best-effort).
    Otherwise the raw text is wrapped as a single free-form claim with
    ``confidence=low``. Never raises: a parsing failure must degrade, not
    kill the round.
    """
    if not structured_output_enabled():
        return _freeform(role, text, default_as_of)

    raw = str(text or "").strip()
    if not raw:
        return AnalystOutput(role=role, as_of=default_as_of, confidence="low", unknowns=["子代理无返回"])

    json_text = _extract_json_object(raw)
    if json_text is not None:
        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and data.get("role"):
                return AnalystOutput.model_validate(data)
        except Exception:
            pass

    return _freeform(role, raw, default_as_of)


def _freeform(role: str, text: str, default_as_of: str | None) -> AnalystOutput:
    content = re.sub(r"```json|```", "", text).strip()
    return AnalystOutput(
        role=role,
        as_of=default_as_of,
        claims=[Claim(text=content, kind="inference")],
        confidence="low",
        unknowns=["子代理未输出结构化 JSON，已按自由文本降级处理"],
    )


def analyst_output_to_text(output: AnalystOutput) -> str:
    """Serialize a structured output back to the text the graph expects.

    When the sub-agent emitted structured JSON, its claim text is returned.
    When it fell back to freeform (or structured mode is off), the original
    text is preserved byte-for-byte so existing pipeline output is unchanged;
    structured fields live only in state.
    """
    if not output.claims:
        return ""
    parts = [claim.text for claim in output.claims if claim.text]
    return "\n\n".join(parts).strip()


def parse_analyst_outputs(role: str, text: str, default_as_of: str | None = None) -> tuple[str, AnalystOutput]:
    """Convenience: parse and return (text_to_store, structured_output)."""
    structured = parse_analyst_output(role, text, default_as_of)
    return analyst_output_to_text(structured), structured