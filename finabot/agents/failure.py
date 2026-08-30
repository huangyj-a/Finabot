"""Eval failure injection (评估报告消融: 随机子 Agent 失败).

A graph node marked to fail returns a structured placeholder instead of
running, so the harness can measure how multi-agent orchestration degrades
under a random sub-agent failure. Controlled via ``FINABOT_EVAL_FAIL_NODE``
(single node name, or ``all``); unset means no injection.
"""

from __future__ import annotations

import os


def injected_failure(node_name: str) -> str | None:
    """Return a failure placeholder if this node is marked to fail, else None.

    The placeholder text signals a degraded-confidence outcome, matching the
    loop-layer timeout placeholder semantics (``[subagent_timeout:...]``).
    """
    target = os.getenv("FINABOT_EVAL_FAIL_NODE", "").strip()
    if not target:
        return None
    if target == "all" or target == node_name:
        return (
            f"[eval_failure:{node_name}] 该节点被注入失败（消融测试），"
            f"其分析缺失，最终结论置信度应相应降级。"
        )
    return None