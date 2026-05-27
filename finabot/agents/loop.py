import asyncio
from openai import AsyncOpenAI
from finabot.bus.queue import MessageBus
from finabot.bus.events import InboundMessage, OutboundMessage
from finabot.agents.session import SessionManager
from finabot.agents.context import ContextBuilder
from finabot.tools.registry import ToolRegistry


class AgentCore:
    def __init__(
        self,
        bus: MessageBus,
        openai_api_key: str,
        model: str = "gpt-3.5-turbo",
    ):
        self.bus = bus
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model

        # 初始化三大模块
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder()
        self.tool_registry = ToolRegistry()  # 后续可以注入 create_tool_registry()

    async def _handle_message(self, msg: InboundMessage):
        """处理单条用户消息"""
        session_key = msg.session_key

        # 1. 保存用户消息到会话
        self.session_manager.add_message(session_key, "user", msg.content)

        # 2. 构建上下文
        history = self.session_manager.get_messages(session_key)
        messages = self.context_builder.build(msg, history)

        # 3. 调用大模型（带工具调用支持）
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tool_registry.get_schemas() or None,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        # 4. 处理工具调用
        if tool_calls:
            messages.append(choice.message.model_dump())
            for tool_call in tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = await self.tool_registry.execute(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result
                })

            # 工具结果返回后，再让模型生成最终回复
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            choice = response.choices[0]

        # 5. 保存助手回复
        assistant_reply = choice.message.content
        self.session_manager.add_message(session_key, "assistant", assistant_reply)

        # 6. 发送回复到消息总线
        outbound_msg = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=assistant_reply
        )
        await self.bus.publish_outbound(outbound_msg)

    async def run(self):
        """Agent 主循环：消费消息总线的 inbound 队列"""
        while True:
            msg = await self.bus.consume_inbound()
            asyncio.create_task(self._handle_message(msg))