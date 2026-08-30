"""真实 LLM 小批量评测：多智能体交叉验证路径相对"基线单次回答"的幻觉率。

方法：
- 每例给定同一份"金标准"工具数据（确定性、含明确数字/来源/日期）。
- 基线：单次 LLM 直接回答（简单提示词，不强制引用数据、允许自由发挥）。
- 多智能体：复现流水线核心——news 分析 → bull∥bear 并行交叉验证 → summary_manager
  综合（使用真实提示词与引用约束），输入均注入同一份金标准数据（不拉真实行情）。
- 校验：确定性规则抽取回答中的数值/百分比/价格/PE/日期/来源，逐一比对是否出现在
  金标准数据中；未出现的计为"无支撑论断（幻觉）"。

注意：这是小样本（可调 EVAL_CASES）的方向性估计，不等于在生产金标准集上的严格指标。
用法：python scripts/eval_hallucination.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from litellm import acompletion

from finabot.agents.llm import get_llm_settings
from finabot.agents.managers.manager import _SUMMARY_MANAGER_PROMPT
from finabot.agents.analysts.news_analyst import _NEWS_ANALYST_PROMPT
from finabot.agents.researchers.bull_researcher import _BULL_RESEARCHER_PROMPT
from finabot.agents.researchers.bear_researcher import _BEAR_RESEARCHER_PROMPT

GOLD_CASES = [
    {
        "name": "贵州茅台",
        "question": "贵州茅台当前适合持有吗？给出区间与依据。",
        "gold": (
            "=== 工具数据（金标准，勿编造）===\n"
            "stock_snapshot: 最新收盘价 1680.00 元，最新交易日 2026-03-06\n"
            "stock_valuation: PE(TTM) 25.3，PB 8.1，PE 处于近5年 40% 分位\n"
            "stock_financial_indicators: 净利润同比 +15.2%，营收同比 +12.0%，毛利率 91.5%\n"
            "stock_fund_flow: 主力净流入 5.6 亿元\n"
            "stock_news: 无直接个股新闻（news_scope=无）\n"
            "来源：东方财富；日期：2026-03-06"
        ),
    },
    {
        "name": "新易盛",
        "question": "新易盛还能继续持有吗？给出风险与依据。",
        "gold": (
            "=== 工具数据（金标准，勿编造）===\n"
            "stock_snapshot: 最新收盘价 320.50 元，最新交易日 2026-03-06\n"
            "stock_valuation: PE(TTM) 48.2，PB 6.9\n"
            "stock_financial_indicators: 营收同比 +40.1%，研发费用 +35.0%，毛利率 30.2%\n"
            "stock_news: 海外 800G 光模块需求旺盛，行业景气（news_scope=market_general）\n"
            "来源：东方财富；日期：2026-03-06"
        ),
    },
]

_BASELINE_SYSTEM = (
    "你是股票投资顾问。根据用户问题直接给出分析结论。"
    "你可以结合自己的常识、行业观点和主观判断来充实回答，不必局限于给定数据。"
)

# ---- 校验规则 ----
# "数据类"论断（公司基本面数据，编造即幻觉）：增速%、PE/PB、日期、资金流规模。
# 价格目标/区间（"X元"）是提示词要求的预测输出，不属于编造数据，故豁免。
_DATA_CLAIM_PATTERNS = [
    re.compile(r"\d{1,3}(?:\.\d+)?\s*%"),              # 增速/占比
    re.compile(r"[Pp][Ee][:：]?\s*\d+\.?\d*"),          # PE
    re.compile(r"[Pp][Bb][:：]?\s*\d+\.?\d*"),          # PB
    re.compile(r"20\d{2}[-/年]\d{1,2}[-/月]\d{1,2}"),   # 日期
    re.compile(r"\d[\d,]*\.?\d*\s*亿(?:元)?"),          # 资金流/市值（亿/亿元）
]
_SOURCE_PATTERN = re.compile(r"东方财富|巨潮资讯|深交所互动易|Wind|通达信|Omdia|中国通信院|上交所|深交所|交易所")


def _normalize(text: str) -> str:
    text = re.sub(r"[\s,，]+", "", text)
    text = text.replace("：", ":").replace("％", "%")
    return text


def extract_claims(answer: str) -> list[str]:
    claims = []
    for pattern in _DATA_CLAIM_PATTERNS:
        for match in pattern.findall(answer):
            claims.append(_normalize(match))
    for match in _SOURCE_PATTERN.findall(answer):
        claims.append(f"来源:{_normalize(match)}")
    return claims


def unsupported_claims(answer: str, gold: str) -> list[str]:
    gold_norm = _normalize(gold)
    unsupported = []
    for claim in extract_claims(answer):
        if not claim:
            continue
        if claim not in gold_norm:
            unsupported.append(claim)
    return unsupported


async def call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    settings = get_llm_settings()
    kwargs = {
        "model": settings.litellm_model,
        "api_key": settings.api_key,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "timeout": settings.timeout_seconds,
    }
    if settings.api_base:
        kwargs["api_base"] = settings.api_base
    response = await acompletion(**kwargs)
    return str(response.choices[0].message.content or "")


_CACHE_DIR = Path(__file__).resolve().parent / ".eval_cache"


async def _cached_answer(name: str, condition: str, producer) -> str:
    """磁盘缓存：同一 (case, condition) 的回答只调一次 LLM，便于调整校验器复验。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}_{condition}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    answer = await producer()
    path.write_text(answer, encoding="utf-8")
    return answer


async def baseline_answer(case: dict) -> str:
    async def _produce() -> str:
        user = f"用户问题：{case['question']}\n\n{case['gold']}\n\n请给出你的分析结论。"
        return await call_llm(_BASELINE_SYSTEM, user)

    return await _cached_answer(case["name"], "baseline", _produce)


async def multi_agent_answer(case: dict) -> str:
    q = case["question"]
    gold = case["gold"]

    async def _news() -> str:
        return await call_llm(
            _NEWS_ANALYST_PROMPT,
            f"用户问题：{q}\n\n=== 数据 ===\n{gold}\n\n请生成新闻/信息线索分析报告。",
        )

    news_report = await _cached_answer(case["name"], "news", _news)

    async def _bull() -> str:
        return await call_llm(
            _BULL_RESEARCHER_PROMPT,
            f"用户问题：{q}\n\n新闻分析师报告：\n{news_report}\n\n请给出看涨论证。",
        )

    async def _bear() -> str:
        return await call_llm(
            _BEAR_RESEARCHER_PROMPT,
            f"用户问题：{q}\n\n新闻分析师报告：\n{news_report}\n\n请给出看跌/风险论证。",
        )

    bull_report, bear_report = await asyncio.gather(
        _cached_answer(case["name"], "bull", _bull),
        _cached_answer(case["name"], "bear", _bear),
    )

    async def _summary() -> str:
        summary_input = (
            f"用户问题：{q}\n\n"
            f"=== 股票基本数据 ===\n{gold}\n\n"
            f"=== 新闻分析数据 ===\n{news_report}\n\n"
            f"=== 看涨研究数据 ===\n{bull_report}\n\n"
            f"=== 看跌研究数据 ===\n{bear_report}\n\n"
            f"请整合以上信息，按标准格式输出最终分析。若某类数据缺失，写'暂无数据'，不要编造。"
        )
        return await call_llm(_SUMMARY_MANAGER_PROMPT, summary_input)

    return await _cached_answer(case["name"], "summary", _summary)


async def run_case(case: dict) -> dict:
    base = await baseline_answer(case)
    multi = await multi_agent_answer(case)
    base_bad = unsupported_claims(base, case["gold"])
    multi_bad = unsupported_claims(multi, case["gold"])
    return {
        "name": case["name"],
        "base_claims": len(extract_claims(base)),
        "base_bad": base_bad,
        "multi_claims": len(extract_claims(multi)),
        "multi_bad": multi_bad,
    }


async def main() -> None:
    cases = GOLD_CASES[: max(1, int(os.getenv("EVAL_CASES", "2")))]
    results = []
    for case in cases:
        print(f"… 评测 {case['name']}（基线 + 多智能体）", file=sys.stderr, flush=True)
        try:
            results.append(await run_case(case))
        except Exception as exc:
            print(f"   {case['name']} 失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if not results:
        print("无成功结果。")
        return

    base_total = sum(r["base_claims"] for r in results)
    base_bad = sum(len(r["base_bad"]) for r in results)
    multi_total = sum(r["multi_claims"] for r in results)
    multi_bad = sum(len(r["multi_bad"]) for r in results)

    print("\n=== 幻觉评测结果（无支撑论断 / 总论断） ===")
    for r in results:
        print(
            f"{r['name']}: 基线 {len(r['base_bad'])}/{r['base_claims']}  "
            f"多智能体 {len(r['multi_bad'])}/{r['multi_claims']}"
        )
        if r["base_bad"]:
            print(f"   基线无支撑: {r['base_bad']}")
        if r["multi_bad"]:
            print(f"   多智能体无支撑: {r['multi_bad']}")
    base_rate = base_bad / base_total * 100 if base_total else 0.0
    multi_rate = multi_bad / multi_total * 100 if multi_total else 0.0
    print(f"\n汇总：基线幻觉率 {base_bad}/{base_total}={base_rate:.1f}%  "
          f"多智能体幻觉率 {multi_bad}/{multi_total}={multi_rate:.1f}%")
    if base_rate > 0:
        print(f"幻觉率降幅 = {(1 - multi_rate / base_rate) * 100:.1f}%")
    else:
        print("基线无幻觉样本，无法计算降幅（小样本限制）。")
    print("\n注：小样本、单一金标准集、规则校验器；非生产级指标。")


if __name__ == "__main__":
    asyncio.run(main())
