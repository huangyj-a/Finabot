"""Cross-turn rolling conversation summary.

Long conversations get their middle history truncated by the context compressor.
Instead of re-truncating the same raw text every turn, this module periodically
asks the LLM to summarise the newly-grown middle section, merges it with the
previous summary, and persists the result in working memory. Next turn the
summary is injected into the prompt memories, so the model keeps continuity even
after the raw middle messages are dropped.

Feature is configurable via env vars (all read lazily at call time):
- FINABOT_ROLLING_SUMMARY             on/off (default on)
- FINABOT_ROLLING_SUMMARY_MIN_MESSAGES    12
- FINABOT_ROLLING_SUMMARY_WINDOW          8
- FINABOT_ROLLING_SUMMARY_TAIL_KEEP       6
- FINABOT_ROLLING_SUMMARY_MAX_CHARS       350
- FINABOT_ROLLING_SUMMARY_MAX_MSG_CHARS   400
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.messages import BaseMessage

from finabot.agents.memory import load_working_memory, save_working_memory

logger = logging.getLogger(__name__)

_WORKING_KEY = "rolling_summary"

# 默认阈值：少于 MIN_MESSAGES 条不生成摘要；中段新增至少 WINDOW 条才更新一次；
# 尾部 TAIL_KEEP 条消息仍在上下文窗口内，不进摘要（避免与最近上下文重复）。
MIN_MESSAGES = 12
WINDOW = 8
TAIL_KEEP = 6
_MAX_MSG_CHARS = 400
_MAX_SUMMARY_CHARS = 350

SUMMARY_SYSTEM_PROMPT = """你是 Finabot 的会话摘要器。把用户与多智能体金融分析助手的对话压缩为结构化中文摘要，必须保留：
- 用户的问题与投资偏好/风险风格
- 关键数据点：股票、价格、估值、财务指标、新闻结论、资金流向等具体数字
- 看多/看空双方的核心论点
- 最终结论与建议
丢弃寒暄、重复与无关内容。直接输出摘要正文，不要任何前缀或解释。"""

SummaryLLM = Callable[[str], Awaitable[str]]


def _internal_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _internal_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or str(default))
    except (TypeError, ValueError):
        return default
    return max(value, 1)


@dataclass(frozen=True)
class RollingSummaryConfig:
    """滚动摘要的开关与阈值；可通过环境变量覆盖。"""

    enabled: bool = True
    min_messages: int = MIN_MESSAGES
    window: int = WINDOW
    tail_keep: int = TAIL_KEEP
    max_summary_chars: int = _MAX_SUMMARY_CHARS
    max_msg_chars: int = _MAX_MSG_CHARS

    @classmethod
    def from_env(cls) -> "RollingSummaryConfig":
        return cls(
            enabled=_internal_env_bool("FINABOT_ROLLING_SUMMARY", True),
            min_messages=_internal_env_int("FINABOT_ROLLING_SUMMARY_MIN_MESSAGES", MIN_MESSAGES),
            window=_internal_env_int("FINABOT_ROLLING_SUMMARY_WINDOW", WINDOW),
            tail_keep=_internal_env_int("FINABOT_ROLLING_SUMMARY_TAIL_KEEP", TAIL_KEEP),
            max_summary_chars=_internal_env_int("FINABOT_ROLLING_SUMMARY_MAX_CHARS", _MAX_SUMMARY_CHARS),
            max_msg_chars=_internal_env_int("FINABOT_ROLLING_SUMMARY_MAX_MSG_CHARS", _MAX_MSG_CHARS),
        )


def _load_state(session_key: str) -> dict[str, Any]:
    data = load_working_memory(session_key) or {}
    state = data.get(_WORKING_KEY) or {}
    if not isinstance(state, dict):
        state = {}
    state.setdefault("summary", "")
    try:
        state["last_summarized_at"] = int(state.get("last_summarized_at", 0) or 0)
    except (TypeError, ValueError):
        state["last_summarized_at"] = 0
    return state


def _save_state(session_key: str, state: dict[str, Any]) -> None:
    data = load_working_memory(session_key) or {}
    data[_WORKING_KEY] = state
    save_working_memory(session_key, data)


def get_rolling_summary(
    session_key: str,
    config: RollingSummaryConfig | None = None,
) -> str:
    """读取该会话已沉淀的滚动摘要；未开启或没有则返回空串。"""
    config = config or RollingSummaryConfig.from_env()
    if not config.enabled:
        return ""
    return _load_state(session_key).get("summary", "")


def _format_messages(
    messages: list[BaseMessage],
    config: RollingSummaryConfig,
) -> str:
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "type", "unknown")
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if len(content) > config.max_msg_chars:
            content = content[: config.max_msg_chars] + "..."
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _should_update(
    messages: list[BaseMessage],
    state: dict[str, Any],
    config: RollingSummaryConfig,
) -> bool:
    if len(messages) < config.min_messages:
        return False
    start = state.get("last_summarized_at", 0)
    if config.tail_keep:
        new_middle = messages[start:-config.tail_keep]
    else:
        new_middle = messages[start:]
    return len(new_middle) >= config.window


async def update_rolling_summary(
    session_key: str,
    messages: list[BaseMessage],
    llm_fn: SummaryLLM,
    config: RollingSummaryConfig | None = None,
) -> bool:
    """中段新增足够多消息时，用 LLM 生成/合并滚动摘要并持久化；返回是否更新。

    未开启或摘要失败均返回 False，不阻断主流程；失败下一轮会重试。
    """
    config = config or RollingSummaryConfig.from_env()
    if not config.enabled:
        return False

    state = _load_state(session_key)
    if not _should_update(messages, state, config):
        return False

    start = state.get("last_summarized_at", 0)
    if config.tail_keep:
        chunk = messages[start:-config.tail_keep]
    else:
        chunk = messages[start:]
    existing = state.get("summary", "")

    if existing:
        prompt = (
            f"这是已有的历史摘要：\n{existing}\n\n"
            f"请把下面新增的对话合并进摘要：保留原摘要要点，新增部分压缩追加，"
            f"总长度仍不超过 {config.max_summary_chars} 字，直接输出合并后的摘要正文：\n\n"
            f"{_format_messages(chunk, config)}"
        )
    else:
        prompt = (
            f"请总结以下对话，输出不超过 {config.max_summary_chars} 字的中文摘要：\n\n"
            f"{_format_messages(chunk, config)}"
        )

    try:
        summary = await llm_fn(prompt)
    except Exception as exc:  # 网络/限流等：不阻断主流程
        logger.warning("rolling summary update failed for %s: %s", session_key, exc)
        return False

    text = str(summary or "").strip()
    if not text:
        return False

    state["summary"] = text
    state["last_summarized_at"] = max(start, len(messages) - config.tail_keep)
    _save_state(session_key, state)
    return True
