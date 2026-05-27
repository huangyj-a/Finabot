"""
Entry point for running finabot as a module: python -m finabot
"""

import asyncio
from finabot.bus.queue import MessageBus
from finabot.agents.loop import AgentCore
from finabot.channels.base import CLIChannel
from finabot.tools.registry import create_tool_registry

# 替换成你的 OpenAI Key（或兼容 OpenAI 的 API）
OPENAI_API_KEY = "sk-xxx"


async def main():
    # 1. 创建消息总线
    bus = MessageBus()

    # 2. 创建 Agent
    agent = AgentCore(bus, openai_api_key=OPENAI_API_KEY)
    # 注入工具注册表
    agent.tool_registry = create_tool_registry()

    # 3. 创建 CLI 通道
    cli_channel = CLIChannel(bus)

    # 4. 同时启动 Agent 和 CLI
    await asyncio.gather(
        agent.run(),
        cli_channel.run()
    )


if __name__ == "__main__":
    asyncio.run(main())