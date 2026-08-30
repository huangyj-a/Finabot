"""离线、确定性测算：持仓分析流水线（单次调度 + 内部多空并行）相对"旧的多跳
supervisor 辩论"在 token 消耗上的节省。

这不是运行时的真实测量，而是对两条**控制流**做精确的 token 记账：
- 旧流程：supervisor 在 fundamental → news → bull → bear → summary 每步之间
  都回 supervisor 重新决策（每跳重读已累积的报告），共 6 次 supervisor 调用；
- 新流程：supervisor 只做 1 次路由 + 1 次最终合成，辩论在流水线内部完成，
  bull/bear 并行、各自只读自己的提示词 + 所需报告（不重读整个对话）。

token 估算采用代码库口径：`ContextCompressorConfig.chars_per_token = 4`。
报告规模等参数可调（env 覆盖），默认取代表性数值。

用法：python scripts/eval_token_reduction.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finabot.agents.llm import SYSTEM_PROMPT
from finabot.agents.managers.manager import _SUMMARY_MANAGER_PROMPT
from finabot.agents.analysts.fundamental_analyst import _FUNDAMENTAL_ANALYST_PROMPT
from finabot.agents.analysts.news_analyst import _NEWS_ANALYST_PROMPT
from finabot.agents.researchers.bull_researcher import _BULL_RESEARCHER_PROMPT
from finabot.agents.researchers.bear_researcher import _BEAR_RESEARCHER_PROMPT

CHARS_PER_TOKEN = 4  # 与 ContextCompressorConfig.chars_per_token 一致


def _tokens(text: str) -> int:
    return max(len(text) // CHARS_PER_TOKEN, 1)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---- 可调参数 ----
QUESTION_TOKENS = _env_int("EVAL_QUESTION_TOKENS", 40)      # 用户问题
HISTORY_TOKENS = _env_int("EVAL_HISTORY_TOKENS", 300)        # 多轮历史
DATA_TOKENS = _env_int("EVAL_DATA_TOKENS", 600)              # 每次行情/新闻数据抓取
SUPERVISOR_ROUTING_OUTPUT = _env_int("EVAL_ROUTING_OUTPUT_TOKENS", 40)
REPORTS = {
    "fundamental": _env_int("EVAL_REPORT_FUNDAMENTAL", 900),
    "news": _env_int("EVAL_REPORT_NEWS", 900),
    "bull": _env_int("EVAL_REPORT_BULL", 700),
    "bear": _env_int("EVAL_REPORT_BEAR", 700),
    "summary": _env_int("EVAL_REPORT_SUMMARY", 1000),
}

SYS_TOKENS = _tokens(SYSTEM_PROMPT)
PROMPT_TOKENS = {
    "fundamental": _tokens(_FUNDAMENTAL_ANALYST_PROMPT),
    "news": _tokens(_NEWS_ANALYST_PROMPT),
    "bull": _tokens(_BULL_RESEARCHER_PROMPT),
    "bear": _tokens(_BEAR_RESEARCHER_PROMPT),
    "summary": _tokens(_SUMMARY_MANAGER_PROMPT),
}


def _sub_agent_input(name: str) -> int:
    """子代理单次调用的输入 token（自己的提示词 + 问题 + 所需上下文/数据）。"""
    if name == "fundamental":
        return PROMPT_TOKENS["fundamental"] + QUESTION_TOKENS + DATA_TOKENS
    if name == "news":
        return PROMPT_TOKENS["news"] + QUESTION_TOKENS + DATA_TOKENS
    if name in {"bull", "bear"}:
        return PROMPT_TOKENS[name] + QUESTION_TOKENS + REPORTS["news"]
    if name == "summary":
        return (
            PROMPT_TOKENS["summary"]
            + QUESTION_TOKENS
            + REPORTS["news"]
            + REPORTS["bull"]
            + REPORTS["bear"]
            + REPORTS["fundamental"]
        )
    raise ValueError(name)


def _old_flow() -> list[tuple[str, int]]:
    """旧流程：supervisor 在每一步之间重新决策，逐跳重读累积报告。"""
    calls: list[tuple[str, int]] = []
    accumulated = 0
    for step in ["fundamental", "news", "bull", "bear", "summary"]:
        calls.append(("supervisor(routing)", SYS_TOKENS + QUESTION_TOKENS + HISTORY_TOKENS + accumulated + SUPERVISOR_ROUTING_OUTPUT))
        calls.append((step, _sub_agent_input(step) + REPORTS[step]))
        accumulated += REPORTS[step]
    # 最终合成
    calls.append(("supervisor(final)", SYS_TOKENS + QUESTION_TOKENS + HISTORY_TOKENS + accumulated + REPORTS["summary"]))
    return calls


def _new_flow() -> list[tuple[str, int]]:
    """新流程：流水线一次调度，内部跑完 5 个子代理；最终 supervisor 只读 summary。"""
    calls: list[tuple[str, int]] = []
    calls.append(("supervisor(routing)", SYS_TOKENS + QUESTION_TOKENS + HISTORY_TOKENS + SUPERVISOR_ROUTING_OUTPUT))
    for step in ["fundamental", "news", "bull", "bear", "summary"]:
        calls.append((step, _sub_agent_input(step) + REPORTS[step]))
    # 最终合成：父图 messages 里只有流水线返回的一条 summary 消息
    calls.append(("supervisor(final)", SYS_TOKENS + QUESTION_TOKENS + HISTORY_TOKENS + REPORTS["summary"]))
    return calls


def _report(title: str, calls: list[tuple[str, int]]) -> int:
    total = sum(t for _, t in calls)
    print(f"\n=== {title} ===")
    for name, t in calls:
        print(f"  {name:<24} {t:>8} tokens")
    print(f"  {'TOTAL':<24} {total:>8} tokens")
    return total


def main() -> None:
    print(f"参数：sys_prompt={SYS_TOKENS}t, question={QUESTION_TOKENS}t, history={HISTORY_TOKENS}t, data={DATA_TOKENS}t")
    print(f"报告规模：{REPORTS}")

    old_total = _report("旧流程：逐跳 supervisor 辩论（11 次 LLM 调用）", _old_flow())
    new_total = _report("新流程：持仓分析流水线（7 次 LLM 调用，多空并行）", _new_flow())

    saved = old_total - new_total
    ratio = saved / old_total * 100 if old_total else 0
    print(f"\nToken 节省：{saved:,} tokens（{ratio:.1f}%）")
    print(f"LLM 调用次数：{len(_old_flow())} -> {len(_new_flow())}")
    print("\n注：节省主要来自省掉的 4 次 supervisor 路由调用（每次都重读已累积报告），"
          "以及最终 supervisor 不再重读 news/bull/bear 全文。")


if __name__ == "__main__":
    main()
