import asyncio
from langchain_core.messages import HumanMessage
from finabot.bus.queue import MessageBus
from finabot.bus.events import InboundMessage, OutboundMessage
from finabot.graph.graph import build_graph
from finabot.agents.session import SessionManager

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

    async def process(self, msg: InboundMessage):
        key = msg.session_key

        expired_keys = self.session_manager.cleanup_expired()
        for expired_key in expired_keys:
            self.sessions.pop(expired_key, None)

        if key not in self.sessions:
            self.sessions[key] = {"messages": [], "session_key": key}

        state = self.sessions[key]
        state["messages"].append(HumanMessage(content=msg.content))
        self.session_manager.add_message(key, "user", msg.content)

        # 运行 LangGraph
        final = await self.graph.ainvoke(state)
        self.sessions[key] = final

        reply = final["messages"][-1].content
        self.session_manager.add_message(key, "assistant", reply)

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