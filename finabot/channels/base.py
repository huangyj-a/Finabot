import asyncio
from finabot.bus.queue import MessageBus
from finabot.bus.events import InboundMessage


class CLIChannel:
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.channel_name = "cli"
        self.chat_id = "cli_chat_001"

    async def input_loop(self):
        """读取用户输入，发布到消息总线"""
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            msg = InboundMessage(
                channel=self.channel_name,
                sender_id="cli_user",
                chat_id=self.chat_id,
                content=user_input
            )
            await self.bus.publish_inbound(msg)

    async def output_loop(self):
        """消费消息总线的 outbound 队列，打印回复"""
        while True:
            msg = await self.bus.consume_outbound()
            print(f"🤖 {msg.content}")

    async def run(self):
        """同时启动输入和输出协程"""
        await asyncio.gather(
            self.input_loop(),
            self.output_loop()
        )