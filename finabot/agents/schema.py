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
    # 默认关闭：生产路径保持自由文本，评估时设 FINABOT_STRUCTURED_OUTPUT=1 显式开启。
    value = os.getenv("FINABOT_STRUCTURED_OUTPUT", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def structured_output_instruction(role: str) -> str:
    """Return the JSON-output instruction to append to a sub-agent prompt.

    The model is asked to emit its normal prose analysis FIRST, then append a
    JSON object. The parser keeps the prose for downstream agents and extracts
    the JSON into structured state (claims/evidence/risk_flags). When the model
    still returns free text without JSON, the tolerant parser degrades.
    """
    return f"""
输出格式要求：先输出你的分析正文（保持原有结构与引用规范），正文之后另起一段，附上一个 JSON 对象用于结构化交接，字段如下（role 固定为 "{role}"）：
{{
  "role": "{role}",
  "as_of": "数据截止日期，无则 null",
  "claims": [
    {{"text": "一条主张", "source_ids": ["引用来源ID或空数组"], "kind": "fact|calculation|inference|opinion", "as_of": "该主张的日期或 null"}}
  ],
  "evidence": ["支撑上述 claims 的证据摘要"],
  "confidence": "high|medium|low",
  "unknowns": ["缺失/不确定的数据或信息"],
  "risk_flags": ["需要提示的风险点"]
}}
JSON 必须是单个合法对象，正文与 JSON 之间不要有其他无关文字。""".strip()


def maybe_append_instruction(role: str, content: str) -> str:
    """Append the JSON-output instruction when structured mode is enabled."""
    if not structured_output_enabled():
        return content
    return f"{content}\n\n{structured_output_instruction(role)}"


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
    """Parse a sub-agent response into ``AnalystOutput`` (pure parser).

    Always tries to extract and validate a JSON object; otherwise wraps the
    raw text as a single free-form claim with ``confidence=low``. Never
    raises. The ``FINABOT_STRUCTURED_OUTPUT`` toggle is NOT applied here — it
    controls instruction injection and state extraction in higher layers
    (``maybe_append_instruction`` / ``parse_subagent_result``).
    """
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


def parse_subagent_result(
    role: str,
    text: str,
    default_as_of: str | None = None,
) -> tuple[str, AnalystOutput]:
    """Split a sub-agent response into (display_text, AnalystOutput).

    When structured mode is on and the response contains a JSON block,
    ``display_text`` is the prose with the JSON removed (so downstream quality
    is unchanged), and the ``AnalystOutput`` carries claims/evidence/risk_flags
    for state. Otherwise ``display_text`` is the full text (freeform degrade).

    Used by graph node wrappers and ``_internal_invoke_sub_agent`` to feed
    structured handoff data into ``AgentState`` without degrading prose.
    """
    raw = str(text or "")
    if not structured_output_enabled():
        return raw, _freeform(role, raw, default_as_of)

    stripped = raw.strip()
    if not stripped:
        return "", AnalystOutput(
            role=role, as_of=default_as_of, confidence="low", unknowns=["子代理无返回"]
        )

    json_text = _extract_json_object(stripped)
    if json_text is not None:
        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and data.get("role"):
                output = AnalystOutput.model_validate(data)
                display = stripped.replace(json_text, "").strip()
                if not display:
                    display = analyst_output_to_text(output)
                return display, output
        except Exception:
            pass

    return stripped, _freeform(role, stripped, default_as_of)


def collect_structured_state(
    role: str,
    text: str,
    default_as_of: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Return (display_text, claims, risk_flags) from a sub-agent response.

    ``claims`` is a list of plain dicts (Pydantic models serialized), suitable
    for accumulation into ``AgentState.claims``. ``risk_flags`` is a list of
    strings. When structured mode is off or the model returned free text, both
    lists are empty (or a single degraded claim) and display_text == text.
    """
    display, structured = parse_subagent_result(role, text, default_as_of)
    if not structured_output_enabled():
        # 关闭时不做结构化抽取，返回原样文本与空增量，避免噪音 claim
        return str(text or ""), [], []
    claims: list[dict[str, Any]] = []
    for claim in structured.claims:
        if hasattr(claim, "model_dump"):
            claims.append(claim.model_dump())
        elif isinstance(claim, dict):
            claims.append(claim)
        else:
            claims.append({"text": str(claim), "kind": "inference"})
    return display, claims, list(structured.risk_flags)


def structured_state_update(
    role: str,
    text: str,
    state: dict[str, Any] | None,
    default_as_of: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return (display_text, state_update) merging claims/risk_flags into state.

    ``display_text`` is the prose (JSON removed) to store in report fields;
    the update dict accumulates ``claims`` and ``risk_flags`` across sub-agents
    (AgentState lists use replace semantics, so merge manually here).
    """
    state = state or {}
    display, claims, risk_flags = collect_structured_state(role, text, default_as_of)
    update: dict[str, Any] = {}
    if claims:
        update["claims"] = list(state.get("claims", []) or []) + claims
    if risk_flags:
        update["risk_flags"] = list(state.get("risk_flags", []) or []) + risk_flags
    return display, update