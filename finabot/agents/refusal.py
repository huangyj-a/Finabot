"""Lightweight compliance refusal classifier (evaluation report: 具体荐股请求).

Non-licensed product boundary: the assistant must NOT give personalized
buy/sell/position advice without compliance conditions being met. This
module deterministically classifies user questions into risk levels; the
supervisor prompt uses the result to force a general-education answer.

It is intentionally rule/keyword based (no extra LLM call): cheap, offline
testable, and auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RefusalDecision:
    level: str  # "safe" | "caution" | "refuse_specific_advice"
    reason: str
    matched_terms: tuple[str, ...] = ()


# 具体买卖/仓位/收益承诺类触发词：命中即拒绝个性化建议，转为一般教育
_SPECIFIC_ADVICE_TERMS = (
    "买入", "卖出", "加仓", "减仓", "清仓", "重仓", "满仓",
    "现在买", "该买", "可以买", "能买", "该卖", "该持有",
    "买多少", "买几成", "仓位建议", "仓位多少", "持仓比例",
    "收益保证", "稳赚", "保证收益", "翻倍", "一定能涨",
)

# 收益承诺 / 确定性用语（加强触发）
_GUARANTEE_TERMS = ("稳赚", "保证收益", "一定涨", "包赚", "无风险收益", "翻倍")

# 一般咨询 / 研究类用语：即使含个别触发词，若带这些词则降级为 caution
_RESEARCH_TERMS = (
    "分析", "怎么看", "基本面", "估值", "财报", "研究", "学习",
    "教育", "科普", "指标", "公式", "如何计算", "区别", "对比",
)


def classify_question(question: str) -> RefusalDecision:
    """Classify a user question into a compliance risk level.

    Returns
    -------
    RefusalDecision
        - ``safe``: 普通研究/数据/教育问题
        - ``caution``: 接近边界（如提及买卖但带研究性限定）
        - ``refuse_specific_advice``: 具体买卖/仓位/收益承诺请求
    """
    text = str(question or "").strip()
    if not text:
        return RefusalDecision("safe", "空问题")

    matched_specific = tuple(term for term in _SPECIFIC_ADVICE_TERMS if term in text)
    matched_guarantee = tuple(term for term in _GUARANTEE_TERMS if term in text)
    matched_research = tuple(term for term in _RESEARCH_TERMS if term in text)

    if matched_guarantee:
        return RefusalDecision(
            "refuse_specific_advice",
            "包含收益承诺/确定性保证用语，必须拒绝个性化买卖建议",
            matched_guarantee,
        )

    if matched_specific:
        if matched_research:
            return RefusalDecision(
                "caution",
                "提及买卖/仓位但带有研究性限定，仅提供一般性教育内容",
                matched_specific + matched_research,
            )
        return RefusalDecision(
            "refuse_specific_advice",
            "未满足合规条件的具体买卖/仓位请求，转为一般性教育回答",
            matched_specific,
        )

    return RefusalDecision("safe", "普通研究/数据/教育问题")


REFUSAL_SYSTEM_NOTE = """
合规边界（必须遵守）：
- 你是一个研究与风险教育辅助工具，不构成投资建议，不提供个性化买卖/仓位/收益承诺。
- 当用户要求"具体买入/卖出/仓位/收益保证"且未满足合规条件时：
  1. 明确说明无法提供具体买卖建议；
  2. 转为一堂风险教育/研究方法课（如何看估值、如何计算收益率、如何理解风险）；
  3. 不给出任何具体股票、具体价位、具体仓位比例或收益预期。
- "仅供参考"不能修复实质荐股；不要为了迎合用户而给出具体操作指令。
""".strip()


def maybe_append_refusal_note(question: str, system_prompt: str) -> str:
    """If the question hits the specific-advice boundary, append the
    compliance note to the system prompt; otherwise return unchanged."""
    decision = classify_question(question)
    if decision.level == "safe":
        return system_prompt
    return f"{system_prompt}\n\n{REFUSAL_SYSTEM_NOTE}"