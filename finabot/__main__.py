import asyncio
from dotenv import load_dotenv
from finabot.bus.queue import MessageBus
from finabot.agents.core import Agent
from finabot.channels.base import CLIChannel


from finabot.cli.commands import app

if __name__ == "__main__":
    app()