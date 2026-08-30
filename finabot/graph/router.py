"""Deterministic intent pre-router (rule fast-path before the LLM supervisor).

Design principles (see docs discussion):
- The supervisor LLM should only decide routing when the intent is genuinely
  open-ended. High-frequency, well-bounded questions get a *deterministic*
  fast path here, saving one LLM round and removing misroute risk on the
  canonical cases.
- The router is CONSERVATIVE: it only short-circuits when it is confident;
  anything uncertain falls through to the supervisor. A wrong short-circuit
  (e.g. forcing a non-stock question into the hold pipeline) is worse than a
  missed one.
- Compliance-boundary questions (specific buy/sell/position requests) are
  NEVER short-circuited: they must reach the supervisor so the refusal note
  path applies.

It is intentionally pure keyword/regex (no network, no LLM): cheap, offline
testable, auditable — same style as ``finabot.agents.refusal``.
"""

from __future__ import annotations

import re

from finabot.agents.refusal import classify_question


# 持有/买卖意图触发词：与"股票标的存在"同时命中才进 hold_analysis_pipeline
_HOLD_INTENT_TERMS = (
    "持有", "买入", "卖出", "加仓", "减仓", "清仓", "重仓", "满仓",
    "该不该买", "适不适合", "能不能买", "可以买", "能买", "适合买",
    "值不值得买", "要不要买", "该不该持有", "还能买", "还能拿",
    "继续拿", "继续持有", "持股", "持仓", "要不要卖", "该不该卖", "能卖", "该卖",
)

# 市场走势类意图（不含具体个股）：仅在"未检测到股票标的"时短路到 market_analyst
_MARKET_INTENT_TERMS = (
    "大盘", "板块", "市场走势", "行情走势", "指数走势", "市场情绪",
    "板块轮动", "行业趋势", "宏观", "大盘趋势", "牛熊", "市场分析",
)

# 展开多空辩论：命中则 hold_analysis_pipeline 附带 debate_mode=True
_DEBATE_TERMS = (
    "辩论", "分别", "展开", "多空", "看多", "看空", "正反",
    "为什么看多", "为什么看空", "多头", "空头", "正反观点",
)

# --- 股票标的存在性（离线启发式，不做网络解析） ---
_STOCK_CODE_PATTERN = re.compile(r"(?<!\d)\d{6}(?!\d)")

_STOCK_NAME_SUFFIX_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,8}(?:股份|集团|科技|银行|证券|医药|能源|汽车|电子|"
    r"地产|白酒|新能源|材料|制造|软件|传媒)"
)

# 高频个股名锚点（无公司后缀也能识别的著名标的），可扩展
_COMMON_NAMES = frozenset({
    "茅台", "宁德", "比亚迪", "五粮液", "隆基", "药明", "平安", "招商银行",
    "新易盛", "腾讯", "阿里", "美团", "京东", "百度", "拼多多", "中石油",
    "中石化", "中国移动", "中国电信", "工业富联", "中芯国际", "海天", "美的",
    "格力", "海尔", "恒瑞", "迈瑞", "爱尔眼科",
})


def _has_stock_target(text: str) -> bool:
    if _STOCK_CODE_PATTERN.search(text):
        return True
    if _STOCK_NAME_SUFFIX_PATTERN.search(text):
        return True
    return any(name in text for name in _COMMON_NAMES)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_intent(question: str) -> tuple[str | None, bool]:
    """规则判定路由意图。

    Returns
    -------
    (target, debate_mode)
        - target: "hold_analysis_pipeline" | "market_analyst" | None
          (None 表示交给 LLM supervisor 决定)
        - debate_mode: 仅在命中 hold 且含辩论关键词时为 True
    """
    text = str(question or "").strip()
    if not text:
        return None, False

    # 合规边界（具体买卖/仓位/收益承诺）：必须走 LLM 路径触发合规提示
    if classify_question(text).level != "safe":
        return None, False

    debate = _has_any(text, _DEBATE_TERMS)
    if _has_stock_target(text) and _has_any(text, _HOLD_INTENT_TERMS):
        return "hold_analysis_pipeline", debate

    # 市场级问题（未点名个股）才短路到 market_analyst，避免把
    # "茅台今天涨了多少"这类单股数据问题错误送进市场分析
    if not _has_stock_target(text) and _has_any(text, _MARKET_INTENT_TERMS):
        return "market_analyst", False

    return None, False
