import typer
import asyncio
from contextlib import suppress
from dotenv import load_dotenv

__logo__ = """
🤖 Finabot - Personal AI Assistant
"""

app = typer.Typer(
    name="finabot",
    help=f"{__logo__} Finabot - Personal AI Assistant",
    no_args_is_help=True,
)

@app.command()
def start(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    heartbeat_interval: float = typer.Option(30.0, "--heartbeat-interval", help="Heartbeat write interval in seconds"),
    maintenance_interval: float = typer.Option(300.0, "--maintenance-interval", help="Maintenance task interval in seconds"),
):
    """Start Finabot AI Assistant"""
    from finabot.agents.core import Agent
    from finabot.bus.queue import MessageBus
    from finabot.bus.events import InboundMessage
    from finabot.runtime import RuntimeService

    load_dotenv()
    bus = MessageBus()
    agent = Agent(bus)
    runtime = RuntimeService(
        agent,
        bus,
        heartbeat_interval_seconds=heartbeat_interval,
        maintenance_interval_seconds=maintenance_interval,
    )

    if ":" in session_id:
        channel_name, chat_id = session_id.split(":", 1)
    else:
        channel_name, chat_id = "cli", session_id

    async def send_message(content: str) -> None:
        await bus.publish_inbound(
            InboundMessage(
                channel=channel_name,
                sender_id="cli_user",
                chat_id=chat_id,
                content=content,
            )
        )

    async def wait_for_response() -> None:
        msg = await bus.consume_outbound()
        typer.echo(f"🤖 {msg.content}")

    async def input_loop() -> None:
        typer.echo("🤖 Finabot 已启动，输入 exit 退出\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except EOFError:
                typer.echo()
                break

            if not user_input.strip():
                continue

            if user_input.lower() in {"exit", "quit"}:
                typer.echo("👋 Goodbye!")
                break

            await send_message(user_input)
            await wait_for_response()

    try:
        if message:
            async def run_once():
                await runtime.start()
                agent_task = asyncio.create_task(agent.run())
                try:
                    await send_message(message)
                    await wait_for_response()
                finally:
                    agent_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await agent_task
                    await runtime.stop()

            asyncio.run(run_once())
        else:
            async def run_interactive():
                await runtime.start()
                agent_task = asyncio.create_task(agent.run())
                try:
                    await input_loop()
                finally:
                    agent_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await agent_task
                    await runtime.stop()
            asyncio.run(run_interactive())
    except KeyboardInterrupt:
        typer.echo("\n👋 Goodbye!")

@app.command()
def version():
    """Show version"""
    typer.echo("finabot v0.1.4.post5")

if __name__ == "__main__":
    app()
