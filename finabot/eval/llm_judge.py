"""Isolated LLM judges for the three reasoning dimensions (评估报告: 评分器组合).

Per the report, news reasoning / bear counterargument / multi-agent synthesis
are graded by *isolated* LLM judges (separate calls, frozen prompts), while the
other six quality dimensions stay deterministic. Each judge returns a 0..1
score; a failed judge call is ignored so the harness falls back to the
deterministic marker score for that dimension.

The judges reuse ``litellm_glm_call``'s ``system_prompt`` override, so they are
independent of the supervisor prompt and never share the agent-under-test's
conversation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from finabot.agents.llm import litellm_glm_call
from finabot.eval.graders import LLM_JUDGE_DIMENSIONS


_NEWS_JUDGE_PROMPT = """
你是金融分析质量的新闻推理裁判（LLM Judge）。只评估"新闻推理"这一维度，不评估其他维度。

评分标准（0..1）：
- 1.0：正确建立了"事件 → 传导机制 → 影响"的因果链，使用条件性语言（可能/或/若），明确区分新闻事实与推测，无伪因果。
- 0.7：有事件与影响描述，但机制链不完整或偶有过度断言。
- 0.4：仅罗列新闻标题/事实，未建立影响机制，或存在"同日发生即因果"式断言。
- 0.0：伪造新闻、把通用市场新闻当作个股直接证据，或因果完全错误。

请仅输出一个 JSON 对象：{"score": 0到1之间的小数, "reason": "一句话评分理由"}
""".strip()


_BEAR_JUDGE_PROMPT = """
你是金融分析质量的看空反证裁判（LLM Judge）。只评估"看空与反证"这一维度，不评估其他维度。

评分标准（0..1）：
- 1.0：反方证据真实且与标的直接相关，重要性判断合理；负面证据不足时明确说"暂无强反证"，不强行唱空。
- 0.7：给出了反方观点但重要性判断一般，或部分风险与标的关联弱。
- 0.4：为了凑"看空"而堆砌泛泛风险（大盘、情绪），或忽略明显的反方证据。
- 0.0：捏造风险、强行唱空，或完全缺失风险提示。

请仅输出一个 JSON 对象：{"score": 0到1之间的小数, "reason": "一句话评分理由"}
""".strip()


_SYNTHESIS_JUDGE_PROMPT = """
你是金融分析质量的多 Agent 综合裁判（LLM Judge）。只评估"多 Agent 综合"这一维度，不评估其他维度。

评分标准（0..1）：
- 1.0：支持/反对/未知三类证据都得到保留，冲突被显式指出并给出触发条件，结论可追溯到上游来源。
- 0.7：保留了两类证据但冲突处理不完整，或部分结论缺少来源。
- 0.4：只保留单边观点、冲突丢失，或结论与上游证据脱节。
- 0.0：报告新增了上游不存在的事实/数字，或最高级别风险无解释地消失。

请仅输出一个 JSON 对象：{"score": 0到1之间的小数, "reason": "一句话评分理由"}
""".strip()


_JUDGE_PROMPTS = {
    "news_reasoning": _NEWS_JUDGE_PROMPT,
    "bear_counter": _BEAR_JUDGE_PROMPT,
    "agent_synthesis": _SYNTHESIS_JUDGE_PROMPT,
}


def _internal_format_input(question: str, final_text: str, reports: dict[str, Any] | None) -> str:
    reports = reports or {}
    sections = [f"用户问题：{question}"]
    for key, label in (
        ("market", "市场分析"),
        ("news", "新闻分析"),
        ("bull", "看涨研究"),
        ("bear", "看跌研究"),
        ("fundamentals", "基本面"),
    ):
        if reports.get(key):
            sections.append(f"=== {label}（交接对象）===\n{str(reports[key])[:1500]}")
    sections.append(f"=== 最终回答 ===\n{str(final_text)[:4000]}")
    return "\n\n".join(sections)


def _internal_parse_score(text: str) -> float | None:
    """Extract a 0..1 score from the judge's JSON/text response."""
    if not text:
        return None
    # 1) JSON {"score": ...}
    try:
        start = text.find("{")
        if start >= 0:
            data = json.loads(text[start : text.rfind("}") + 1])
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                return max(0.0, min(score, 1.0))
    except (ValueError, TypeError):
        pass
    # 2) 文本内 "score: 0.8" 或裸小数
    match = re.search(r"(?:score\s*[:=]\s*)?(0(?:\.\d+)?|1(?:\.0+)?)", text, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return max(0.0, min(score, 1.0))
    return None


async def judge_dimension(
    dimension: str,
    question: str,
    final_text: str,
    reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge a single dimension with an isolated LLM call.

    Returns ``{"dimension", "score", "reason"}``. ``score`` is None on any
    error, so callers fall back to the deterministic score.
    """
    prompt = _JUDGE_PROMPTS.get(dimension)
    if prompt is None:
        return {"dimension": dimension, "score": None, "reason": "未知维度"}

    content = _internal_format_input(question, final_text, reports)
    try:
        response = await litellm_glm_call(
            messages=[HumanMessage(content=content)],
            system_prompt=prompt,
            stream_label=None,
        )
        raw = str(getattr(response, "content", "") or "")
    except Exception as exc:  # 网络/限流：不阻断评分，回退确定性
        return {"dimension": dimension, "score": None, "reason": f"judge_error:{type(exc).__name__}"}

    score = _internal_parse_score(raw)
    if score is None:
        return {"dimension": dimension, "score": None, "reason": "judge_parse_failed"}
    return {"dimension": dimension, "score": score, "reason": raw[:200]}


async def judge_quality_dimensions(
    question: str,
    final_text: str,
    reports: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Judge the 3 LLM dimensions in isolation; return only successful scores.

    A failed dimension is omitted (not set to 0), so the harness merges the
    judge scores over the deterministic marker baseline.
    """
    results: dict[str, float] = {}
    for dimension in LLM_JUDGE_DIMENSIONS:
        outcome = await judge_dimension(dimension, question, final_text, reports)
        if outcome.get("score") is not None:
            results[dimension] = outcome["score"]
    return results