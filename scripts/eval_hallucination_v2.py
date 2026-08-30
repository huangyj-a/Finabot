"""严谨版幻觉评测 v2：真实 AKShare 金标准 + 判官 LLM 三类分类 + 多次采样。

设计：
1. 金标准：对每个标的用真实 AKShare 抓取（get_cached_akshare_data + format_akshare_data），
   落盘缓存（.eval_cache_v2/gold/）。某标的抓取失败或数据过少则跳过。
2. 两种条件，给定【同一份金标准数据】：
   - baseline：单次 LLM 直接回答（普通投顾提示词，允许自由发挥）。
   - multi_agent：复现流水线核心 news → bull∥bear（并行）→ summary_manager，
     使用真实子代理提示词与引用约束。
   每个条件独立采样 EVAL_SAMPLES 次（temperature=0.5），回答落盘缓存。
3. 判官：每个回答一次判官 LLM 调用，把论断分类为 data_supported / fabricated / interpretive。
   幻觉率 = fabricated / (fabricated + data_supported)（interpretive 是研判/建议，不计）。
4. 汇总：两种条件的聚合幻觉率与相对降幅。

用法：
  python scripts/eval_hallucination_v2.py
  EVAL_STOCKS="贵州茅台,宁德时代,招商银行,五粮液" EVAL_SAMPLES=2 python scripts/eval_hallucination_v2.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from litellm import acompletion

from finabot.agents.llm import get_llm_settings
from finabot.agents.akshare_cache import format_akshare_data, get_cached_akshare_data
from finabot.agents.hold_pipeline import KEY_SECTIONS
from finabot.agents.managers.manager import _SUMMARY_MANAGER_PROMPT
from finabot.agents.analysts.news_analyst import _NEWS_ANALYST_PROMPT
from finabot.agents.researchers.bull_researcher import _BULL_RESEARCHER_PROMPT
from finabot.agents.researchers.bear_researcher import _BEAR_RESEARCHER_PROMPT

DEFAULT_STOCKS = ["贵州茅台", "宁德时代", "招商银行", "五粮液", "新易盛", "中国平安"]
_CACHE = Path(__file__).resolve().parent / ".eval_cache_v2"
_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.5"))

_BASELINE_SYSTEM = (
    "你是股票投资顾问。根据用户问题直接给出分析结论。"
    "你可以结合自己的常识、行业观点和主观判断来充实回答，不必局限于给定数据。"
)

_JUDGE_SYSTEM = (
    "你是金融回答的幻觉检测器。给定【金标准数据】和【待检测回答】，把回答中所有"
    "【声称具体事实】的论断抽取出来并分类：\n"
    "- data_supported：论断声称的事实能在金标准数据中找到（数值/日期/来源/事件一致）；\n"
    "- fabricated：论断声称了金标准数据中不存在的事实（例如金标准没有的增速、PE、日期、"
    "来源、公司事件、新闻、具体数值）；\n"
    "- interpretive：属于预测/观点/风险判断/建议/区间研判，不声称具体数据事实"
    "（如“预计上涨”“建议观望”“风险较高”“目标价区间”）。\n"
    "只输出 JSON：{\"fabricated\":[...], \"data_supported\":[...], \"interpretive\":[...]}\n"
    "不要输出任何其它文字或代码块标记。"
)


def _cache_path(*parts: str) -> Path:
    return _CACHE.joinpath(*parts)


def _read_cache(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except Exception:
        return None


def _write_cache(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


async def call_llm(system: str, user: str, temperature: float = _TEMPERATURE) -> str:
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


async def fetch_gold(stock: str) -> str:
    """真实 AKShare 抓取并格式化为金标准文本；缓存到磁盘。"""
    cache = _cache_path("gold", f"{stock}.txt")
    cached = _read_cache(cache)
    if cached:
        return cached
    payload = await asyncio.wait_for(
        asyncio.to_thread(get_cached_akshare_data, {}, f"{stock} 股票分析"),
        timeout=180,
    )
    fields = KEY_SECTIONS + ["stock_news"]
    gold = format_akshare_data(payload, fields)
    header = (
        f"标的：{stock}\n"
        f"fetch_time：{payload.get('fetch_time', '未知')}\n"
        f"resolved_symbol：{payload.get('resolved_symbol', '未知')}\n"
    )
    _write_cache(cache, header + "\n" + gold)
    return header + "\n" + gold


def _gold_has_enough(gold: str) -> bool:
    # 至少要有一项实质行情内容，否则金标准太弱，跳过该标的
    return bool(re.search(r"## stock_(spot|snapshot|valuation|financial_indicators)", gold))


async def _cached_llm(name: str, condition: str, system: str, user: str) -> str:
    path = _cache_path("answers", f"{name}_{condition}.txt")
    cached = _read_cache(path)
    if cached is not None:
        return cached
    answer = await call_llm(system, user)
    _write_cache(path, answer)
    return answer


async def baseline_answer(stock: str, gold: str, sample: int) -> str:
    user = f"用户问题：{stock} 当前适合持有吗？给出结论与依据。\n\n=== 数据 ===\n{gold}\n\n请给出你的分析结论。"
    return await _cached_llm(stock, f"baseline_{sample}", _BASELINE_SYSTEM, user)


async def _cached_step(name: str, condition: str, producer) -> str:
    """按 (标的, 步骤) 缓存单次 LLM 产出，避免重复调用。"""
    path = _cache_path("answers", f"{name}_{condition}.txt")
    cached = _read_cache(path)
    if cached is not None:
        return cached
    value = await producer()
    _write_cache(path, value)
    return value


async def multi_agent_answer(stock: str, gold: str, sample: int) -> str:
    user_q = f"用户问题：{stock} 当前适合持有吗？给出结论与依据。"

    news_report = await _cached_step(
        stock,
        f"news_{sample}",
        lambda: call_llm(
            _NEWS_ANALYST_PROMPT,
            f"{user_q}\n\n=== 数据 ===\n{gold}\n\n请生成新闻/信息线索分析报告。",
        ),
    )

    bull_task = _cached_step(
        stock,
        f"bull_{sample}",
        lambda: call_llm(
            _BULL_RESEARCHER_PROMPT,
            f"{user_q}\n\n新闻分析师报告：\n{news_report}\n\n请给出看涨论证。",
        ),
    )
    bear_task = _cached_step(
        stock,
        f"bear_{sample}",
        lambda: call_llm(
            _BEAR_RESEARCHER_PROMPT,
            f"{user_q}\n\n新闻分析师报告：\n{news_report}\n\n请给出看跌/风险论证。",
        ),
    )
    bull, bear = await asyncio.gather(bull_task, bear_task)

    return await _cached_step(
        stock,
        f"summary_{sample}",
        lambda: call_llm(
            _SUMMARY_MANAGER_PROMPT,
            (
                f"{user_q}\n\n"
                f"=== 股票基本数据 ===\n{gold}\n\n"
                f"=== 新闻分析数据 ===\n{news_report}\n\n"
                f"=== 看涨研究数据 ===\n{bull}\n\n"
                f"=== 看跌研究数据 ===\n{bear}\n\n"
                f"请整合以上信息，按标准格式输出最终分析。若某类数据缺失，写'暂无数据'，不要编造。"
            ),
        ),
    )


async def judge_answer(stock: str, gold: str, answer: str, condition: str, sample: int) -> dict:
    path = _cache_path("judge", f"{stock}_{condition}_{sample}.json")
    cached = _read_cache(path)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            pass
    user = (
        f"【金标准数据】\n{gold}\n\n"
        f"【待检测回答】\n{answer}\n\n"
        f"请输出上述 JSON。"
    )
    raw = await call_llm(_JUDGE_SYSTEM, user, temperature=0.0)
    parsed = _parse_judge_json(raw)
    if parsed is None:
        raw = await call_llm(
            _JUDGE_SYSTEM + " 直接输出合法 JSON，不要包含解释。", user, temperature=0.0
        )
        parsed = _parse_judge_json(raw) or {"fabricated": [], "data_supported": [], "interpretive": []}
    _write_cache(path, json.dumps(parsed, ensure_ascii=False))
    return parsed


def _parse_judge_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return {
            "fabricated": list(data.get("fabricated") or []),
            "data_supported": list(data.get("data_supported") or []),
            "interpretive": list(data.get("interpretive") or []),
        }
    return None


async def run_case(stock: str, gold: str, samples: int) -> dict:
    result = {"stock": stock, "baseline": {"fab": 0, "sup": 0}, "multi": {"fab": 0, "sup": 0}}
    for s in range(samples):
        base = await baseline_answer(stock, gold, s)
        multi = await multi_agent_answer(stock, gold, s)
        base_j = await judge_answer(stock, gold, base, "baseline", s)
        multi_j = await judge_answer(stock, gold, multi, "multi", s)
        result["baseline"]["fab"] += len(base_j["fabricated"])
        result["baseline"]["sup"] += len(base_j["data_supported"])
        result["multi"]["fab"] += len(multi_j["fabricated"])
        result["multi"]["sup"] += len(multi_j["data_supported"])
    return result


async def main() -> None:
    stocks = [s.strip() for s in os.getenv("EVAL_STOCKS", ",".join(DEFAULT_STOCKS)).split(",") if s.strip()]
    samples = max(1, int(os.getenv("EVAL_SAMPLES", "2")))

    golds: list[tuple[str, str]] = []
    for stock in stocks:
        print(f"… 抓取金标准 {stock}", file=sys.stderr, flush=True)
        try:
            gold = await fetch_gold(stock)
            if _gold_has_enough(gold):
                golds.append((stock, gold))
            else:
                print(f"   跳过 {stock}：金标准数据过少", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"   跳过 {stock}：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if not golds:
        print("无可用金标准标的。")
        return

    print(f"金标准标的 {len(golds)} 个，每条件采样 {samples} 次", file=sys.stderr, flush=True)

    base_fab = base_sup = multi_fab = multi_sup = 0
    per_stock = []
    for stock, gold in golds:
        print(f"… 评测 {stock}（基线 + 多智能体）", file=sys.stderr, flush=True)
        r = await run_case(stock, gold, samples)
        per_stock.append(r)
        base_fab += r["baseline"]["fab"]
        base_sup += r["baseline"]["sup"]
        multi_fab += r["multi"]["fab"]
        multi_sup += r["multi"]["sup"]

    base_rate = base_fab / (base_fab + base_sup) * 100 if (base_fab + base_sup) else 0.0
    multi_rate = multi_fab / (multi_fab + multi_sup) * 100 if (multi_fab + multi_sup) else 0.0

    print("\n=== 幻觉率（fabricated / (fabricated+data_supported)） ===")
    for r in per_stock:
        b = r["baseline"]
        m = r["multi"]
        br = b["fab"] / (b["fab"] + b["sup"]) * 100 if (b["fab"] + b["sup"]) else 0.0
        mr = m["fab"] / (m["fab"] + m["sup"]) * 100 if (m["fab"] + m["sup"]) else 0.0
        print(f"  {r['stock']}: 基线 {b['fab']}/({b['fab']}+{b['sup']})={br:.1f}%  "
              f"多智能体 {m['fab']}/({m['fab']}+{m['sup']})={mr:.1f}%")
    print(f"\n汇总：基线 {base_fab}/({base_fab}+{base_sup})={base_rate:.1f}%  "
          f"多智能体 {multi_fab}/({multi_fab}+{multi_sup})={multi_rate:.1f}%")
    if base_rate > 0:
        print(f"相对降幅 = {(1 - multi_rate / base_rate) * 100:.1f}%")
    else:
        print("基线幻觉率为 0，无法计算降幅。")
    print(f"\n样本量：{len(golds)} 标的 × {samples} 采样（真实 AKShare 金标准 + 判官 LLM 分类）")


if __name__ == "__main__":
    asyncio.run(main())
