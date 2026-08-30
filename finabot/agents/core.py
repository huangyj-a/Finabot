import asyncio
import logging
import os
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from finabot.bus.queue import MessageBus
from finabot.bus.events import InboundMessage, OutboundMessage
from finabot.graph.graph import build_graph
from finabot.agents.memory import (
    build_memory_context,
    load_short_memory,
    load_working_memory,
    record_run_memory,
    save_short_memory,
    save_working_memory,
)
from finabot.agents.llm import litellm_glm_call
from finabot.agents.rolling_summary import (
    SUMMARY_SYSTEM_PROMPT,
    get_rolling_summary,
    update_rolling_summary,
)
from finabot.agents.streaming import set_token_sink, reset_token_sink


async def _internal_summary_llm(prompt: str) -> str:
    """会话摘要器专用 LLM 调用：覆盖系统提示、不流式。"""
    response = await litellm_glm_call(
        messages=[HumanMessage(content=prompt)],
        stream_label=None,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
    )
    return str(getattr(response, "content", "") or "")


logger = logging.getLogger(__name__)

_RUN_SCOPED_STATE_DEFAULTS = {
    "akshare_cache": dict,
    "market_report": str,
    "news_report": str,
    "bull_report": str,
    "bear_report": str,
    "fundamentals_report": str,
    "debate_context": dict,
    "debate_mode": bool,
    "evidence_registry": dict,
    "claims": list,
    "risk_flags": list,
    "run_meta": dict,
}

# 会话 TTL：与旧 SessionManager 默认一致
_SESSION_TTL = timedelta(minutes=60)


class Agent:
    def __init__(self, bus: MessageBus):
        self.bus = bus
        # LangGraph checkpointer 接管会话状态（messages 等）：
        # 每个 thread_id 一份图状态，跨轮次自动累积，不再用 Agent.sessions 手动维护。
        self.checkpointer = MemorySaver()
        self.graph = build_graph(self.checkpointer)
        # 会话 TTL 追踪（checkpointer 本身不提供过期清理）
        self._thread_last_used: dict[str, datetime] = {}
        self._tasks: set[asyncio.Task] = set()
        self._session_locks: dict[str, asyncio.Lock] = {}

    def _discard_task(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)

    def _report_task_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Unhandled message task failure")

    def _restore_messages(self, history: list[dict]) -> list:
        messages = []
        for item in history:
            role = item.get("role") if isinstance(item, dict) else None
            content = item.get("content", "") if isinstance(item, dict) else ""
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
        return messages

    def _reset_run_state(self, state: dict) -> None:
        """把单轮分析字段重置为初始值（每次调用前应用，避免脏数据串轮）。"""
        for field, factory in _RUN_SCOPED_STATE_DEFAULTS.items():
            state[field] = factory()

    @property
    def active_sessions(self) -> dict[str, datetime]:
        """供 runtime 心跳展示的会话活动信息。"""
        return dict(self._thread_last_used)

    def _thread_config(self, key: str) -> dict:
        return {"configurable": {"thread_id": key}}

    def _invoke_config(self, key: str) -> dict:
        """invoke/astream 配置：thread_id + 显式 recursion_limit（防止无限循环）。"""
        limit = max(1, int(os.getenv("FINABOT_MAX_RECURSION", "16")))
        return {"configurable": {"thread_id": key}, "recursion_limit": limit}

    async def _cleanup_thread(self, key: str) -> None:
        """删除一个会话：checkpointer 线程 + 锁 + TTL 追踪。"""
        self._thread_last_used.pop(key, None)
        lock = self._session_locks.get(key)
        if lock is not None and not lock.locked():
            self._session_locks.pop(key, None)
        try:
            await self.checkpointer.adelete_thread(self._thread_config(key))
        except Exception:
            logger.debug("checkpointer delete_thread failed for %s", key, exc_info=True)

    async def _cleanup_expired(self) -> list[str]:
        """清理超过 TTL 未活动的会话，返回被清理的会话键。"""
        now = datetime.now()
        expired_keys = [
            key for key, last in self._thread_last_used.items()
            if now - last > _SESSION_TTL
        ]
        for key in expired_keys:
            await self._cleanup_thread(key)
        return expired_keys

    async def process(self, msg: InboundMessage):
        key = msg.session_key
        lock = self._session_locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                await self._process_locked(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Failed to process message for session %s", key)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Finabot 处理请求失败，请稍后重试。",
                        metadata={"error": True, "error_type": type(exc).__name__},
                    )
                )

    async def _process_locked(self, msg: InboundMessage):
        key = msg.session_key

        for expired_key in await self._cleanup_expired():
            logger.debug("expired session cleaned: %s", expired_key)

        persistent_history = load_short_memory(key)
        config = self._thread_config(key)

        # 首次（或进程重启后 checkpointer 为空）：用磁盘历史重建 messages；
        # 已有 checkpoint 时 messages 由 checkpointer 承载，只追加本条新消息。
        snapshot = await self.graph.aget_state(config)
        # 注意：无 checkpoint 时 aget_state 返回 values={}（而非 None），
        # 所以用"空值"判断是否首次/重启，而非 values is None。
        restored = self._restore_messages(persistent_history) if not snapshot.values else []

        from finabot.utils.clock import now as clock_now

        # 记忆 = 短期/长期/知识 + 跨轮次滚动摘要（长对话连续性）
        memories = build_memory_context(key, msg.sender_id, msg.content)
        rolling_summary = get_rolling_summary(key)
        if rolling_summary:
            memories.append({"summary": "跨轮次历史摘要", "content": rolling_summary})

        input_state = {
            "messages": restored + [HumanMessage(content=msg.content)],
            "session_key": key,
            "user_id": msg.sender_id,
            "memories": memories,
            "as_of": os.environ.get("FINABOT_EVAL_AS_OF", "").strip() or None,
            "run_meta": {
                "started_at": clock_now().isoformat(),
                "llm_calls": 0,
                "subagent_timeouts": [],
            },
        }
        self._reset_run_state(input_state)
        persistent_history.append({"role": "user", "content": msg.content})
        self._thread_last_used[key] = datetime.now()

        # 运行 LangGraph（流式 + checkpointer）：父图用 astream(subgraphs=True)
        # 透出持有分析子图每一步的产出；可流式标签（fundamental/news/supervisor）
        # 的 LLM 调用通过 token sink 实时推送"打字机"效果。最终状态直接从
        # checkpointer 读取权威快照（含跨轮累积的历史）。
        async def _publish_token(node_label: str, text: str) -> None:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=text,
                    metadata={"stream": "token", "node": node_label},
                )
            )

        sink_token = set_token_sink(_publish_token)
        try:
            async for namespace, chunk in self.graph.astream(input_state, config=self._invoke_config(key), subgraphs=True):
                for _node_name, node_update in chunk.items():
                    if not namespace:
                        # 顶层图节点：最终状态以 checkpointer 快照为准，跳过
                        continue
                    # 子图（持有分析流水线）步骤：
                    # - fetch / bull / bear 以整段进度推送（fetch 无 LLM 输出，
                    #   bull/bear 为并行节点，token 会交错，故用整段更清晰）；
                    # - fundamental / news 由 token sink 流式输出，避免重复；
                    # - summary 跳过，由最终答复统一给出结论。
                    node = namespace[-1]
                    if node == "summary" or node in {"fundamental", "news"}:
                        continue
                    for message in node_update.get("messages") or []:
                        content = getattr(message, "content", "") or ""
                        if content:
                            await self.bus.publish_outbound(
                                OutboundMessage(
                                    channel=msg.channel,
                                    chat_id=msg.chat_id,
                                    content=content,
                                    metadata={"stream": "progress", "node": _node_name},
                                )
                            )
        finally:
            reset_token_sink(sink_token)

        snapshot = await self.graph.aget_state(config)
        final: dict = dict(snapshot.values) if snapshot.values is not None else dict(input_state)

        reply = final["messages"][-1].content
        # 沉淀长期记忆：用户画像（偏好/风险风格）、关注股票、历史分析结论
        # （尽力而为，失败不阻断主流程）
        record_run_memory(
            user_id=msg.sender_id,
            question=msg.content,
            final_state=final,
            reply=reply,
        )
        persistent_history.append({"role": "assistant", "content": reply})
        save_short_memory(key, persistent_history)
        # 合并写入 working memory，避免覆盖滚动摘要等其它模块写入的字段
        working = load_working_memory(key) or {}
        working.update(
            {
                "session_key": key,
                "user_id": msg.sender_id,
                "debate_context": final.get("debate_context", {}),
                "message_count": len(final.get("messages", [])),
                "update_time": datetime.now().isoformat(),
            }
        )
        save_working_memory(key, working)

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=reply,
                metadata={"final": True},
            )
        )

        # 跨轮次滚动摘要：长对话中段定期用 LLM 汇总并持久化，下一轮注入提示词。
        # 尽力而为：失败只记日志，不阻断主流程（也不阻塞下一条消息太久）。
        try:
            await update_rolling_summary(key, final.get("messages", []), _internal_summary_llm)
        except Exception:
            logger.exception("rolling summary update failed for session %s", key)

    async def run(self):
        try:
            while True:
                msg = await self.bus.consume_inbound()
                task = asyncio.create_task(self.process(msg))
                self._tasks.add(task)
                task.add_done_callback(self._discard_task)
                task.add_done_callback(self._report_task_result)
        except asyncio.CancelledError:
            return
