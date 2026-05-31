import asyncio
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage
from finabot.bus.queue import MessageBus
from finabot.bus.events import InboundMessage, OutboundMessage
from finabot.graph.graph import build_graph
from finabot.agents.session import SessionManager
from finabot.agents.memory import build_memory_context, load_short_memory, save_short_memory, save_working_memory

class Agent:
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.graph = build_graph()
        self.sessions = {}
        self.session_manager = SessionManager()
        self._tasks: set[asyncio.Task] = set()

    def _discard_task(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)

    def _report_task_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            print(f"❌ 处理消息时出错: {exc}")

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

    async def process(self, msg: InboundMessage):
        key = msg.session_key

        expired_keys = self.session_manager.cleanup_expired()
        for expired_key in expired_keys:
            self.sessions.pop(expired_key, None)

        persistent_history = load_short_memory(key)
        if key not in self.sessions:
            self.sessions[key] = {
                "messages": self._restore_messages(persistent_history),
                "session_key": key,
                "user_id": msg.sender_id,
                "memories": [],
                "akshare_cache": {},
            }

        state = self.sessions[key]
        state["akshare_cache"] = {}
        state["user_id"] = msg.sender_id
        state["memories"] = build_memory_context(key, msg.sender_id, msg.content)
        state["messages"].append(HumanMessage(content=msg.content))
        self.session_manager.add_message(key, "user", msg.content)
        persistent_history.append({"role": "user", "content": msg.content})

        # 运行 LangGraph
        final = await self.graph.ainvoke(state)
        self.sessions[key] = final

        reply = final["messages"][-1].content
        self.session_manager.add_message(key, "assistant", reply)
        persistent_history.append({"role": "assistant", "content": reply})
        save_short_memory(key, persistent_history)
        save_working_memory(
            key,
            {
                "session_key": key,
                "user_id": msg.sender_id,
                "debate_context": final.get("debate_context", {}),
                "message_count": len(final.get("messages", [])),
                "update_time": datetime.now().isoformat(),
            },
        )

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=reply
            )
        )
    
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
