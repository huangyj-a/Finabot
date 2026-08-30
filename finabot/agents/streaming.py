"""Token-level streaming sink shared by the LLM layer and the agent core.

`litellm_glm_call` checks the current contextvar before each call; if a sink is
installed (set by `Agent.process` while the graph runs), and the call site
declares a streamable label, the completion is made with `stream=True` and every
text delta is forwarded as ``(label, token)`` to the sink. This keeps token
streaming opt-in and out of band of the message bus's normal flow.
"""

from __future__ import annotations

import contextvars
from typing import Awaitable, Callable

# 签名：(node_label, token_text) -> Awaitable[None]
TokenSink = Callable[[str, str], Awaitable[None]]

_token_sink: contextvars.ContextVar[TokenSink | None] = contextvars.ContextVar(
    "finabot_token_sink", default=None
)

# 哪些 LLM 调用点允许流式输出（其余保持整段返回，避免并行节点 token 交错）
_STREAMABLE_LABELS: frozenset[str] = frozenset(
    {"supervisor", "fundamental_analyst", "news_analyst"}
)


def is_streamable_label(label: str | None) -> bool:
    return label in _STREAMABLE_LABELS


def set_token_sink(sink: TokenSink) -> contextvars.Token:
    return _token_sink.set(sink)


def reset_token_sink(token: contextvars.Token) -> None:
    _token_sink.reset(token)


def get_token_sink() -> TokenSink | None:
    return _token_sink.get()
