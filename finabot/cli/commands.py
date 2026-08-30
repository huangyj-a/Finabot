import asyncio
import os
import sys
from contextlib import suppress

import typer
from dotenv import load_dotenv

from finabot import __version__

__logo__ = """
🤖 Finabot - Personal AI Assistant
"""

app = typer.Typer(
    name="finabot",
    help=f"{__logo__} Finabot - Personal AI Assistant",
    no_args_is_help=True,
)


def _safe_echo(text: str, *, err: bool = False, nl: bool = True) -> None:
    """编码安全的输出：在 GBK 等无法表示 emoji 的控制台下回退，避免整轮崩溃。

    nl=False 时不换行，用于 token 打字机内联渲染。
    """
    try:
        typer.echo(text, err=err, nl=nl)
    except UnicodeEncodeError:
        stream = sys.stderr if err else sys.stdout
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = str(text).encode(encoding, "replace").decode(encoding, "replace")
        try:
            typer.echo(safe, err=err, nl=nl)
        except UnicodeEncodeError:
            typer.echo(str(text).encode("utf-8", "ignore").decode("utf-8"), err=err, nl=nl)


def _write_inline(text: str) -> None:
    """不换行直接写到 stdout，供 token 逐片打字机渲染。"""
    try:
        sys.stdout.write(str(text))
        sys.stdout.flush()
    except Exception:
        pass


def _response_timeout_seconds() -> float:
    try:
        value = float(os.getenv("FINABOT_RESPONSE_TIMEOUT_SECONDS", "120"))
    except ValueError as exc:
        raise typer.BadParameter(
            "FINABOT_RESPONSE_TIMEOUT_SECONDS 必须是正数。"
        ) from exc
    if value <= 0:
        raise typer.BadParameter(
            "FINABOT_RESPONSE_TIMEOUT_SECONDS 必须是正数。"
        )
    return value


@app.command()
def start(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    heartbeat_interval: float = typer.Option(30.0, "--heartbeat-interval", help="Heartbeat write interval in seconds"),
    maintenance_interval: float = typer.Option(300.0, "--maintenance-interval", help="Maintenance task interval in seconds"),
):
    """Start Finabot AI Assistant"""
    load_dotenv()

    from finabot.agents.core import Agent
    from finabot.bus.queue import MessageBus
    from finabot.bus.events import InboundMessage
    from finabot.runtime import RuntimeService
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

    response_timeout = _response_timeout_seconds()

    async def wait_for_response() -> None:
        """持续消费 outbound，直到拿到 final 消息；token 内联渲染，进度分块打印。"""
        final_reply: str | None = None
        current_node: str | None = None
        saw_inline = False
        try:
            while final_reply is None:
                try:
                    msg = await asyncio.wait_for(bus.consume_outbound(), timeout=response_timeout)
                except asyncio.TimeoutError:
                    _safe_echo(f"Finabot 响应超时（{response_timeout:g} 秒），请稍后重试。", err=True)
                    return

                meta = msg.metadata or {}
                if meta.get("stream") == "token":
                    node = meta.get("node", "") or ""
                    if node != current_node:
                        if current_node is not None and saw_inline:
                            _safe_echo()
                        if node:
                            _safe_echo(f"[{node}] ", nl=False)
                        current_node = node
                    _write_inline(msg.content)
                    saw_inline = True
                elif meta.get("stream") == "progress":
                    if saw_inline:
                        _safe_echo()
                        saw_inline = False
                    _safe_echo(f"[{meta.get('node', '')}] {msg.content}")
                    current_node = None
                elif meta.get("final") or meta.get("error"):
                    final_reply = msg.content
                else:
                    # 兜底：未标注的消息一律视为最终答复
                    final_reply = msg.content
        finally:
            if saw_inline:
                _safe_echo()
        if final_reply is not None:
            _safe_echo(f"🤖 {final_reply}")

    async def input_loop() -> None:
        _safe_echo("🤖 Finabot 已启动，输入 exit 退出\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except EOFError:
                typer.echo()
                break

            if not user_input.strip():
                continue

            if user_input.lower() in {"exit", "quit"}:
                _safe_echo("👋 Goodbye!")
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
        _safe_echo("\n👋 Goodbye!")

@app.command()
def version():
    """Show version"""
    typer.echo(f"finabot v{__version__}")


@app.command()
def eval_run(
    suite: str = typer.Option("dev", "--suite", help="Task suite: dev | regression | hidden"),
    task_id: str = typer.Option(None, "--task", help="Run only this task id"),
    trials: int = typer.Option(1, "--trials", help="Trials per task"),
    quality_threshold: float = typer.Option(75.0, "--threshold", help="Quality pass threshold"),
    judge: bool = typer.Option(False, "--judge", help="启用隔离 LLM Judge（新闻/反证/综合三维度）"),
):
    """Run the evaluation harness over a task suite (评估实操报告落地).

    使用冻结数据（eval/fixtures/<task_id>/snapshot.json）离线运行真实图；
    无 fixture 的任务回退到实时数据。每任务 trials 次，输出指标汇总。
    需要 LLM 凭据（.env）。--judge 时额外用隔离 LLM Judge 评新闻/反证/综合。
    """
    load_dotenv()

    import asyncio
    import json

    from finabot.eval.harness import EvalRunner
    from finabot.eval.metrics import pass_all_n, summarize_trials
    from finabot.eval.tasks import find_task_root, load_task, load_tasks

    root = find_task_root()
    suite_dir = root.parent / suite
    if task_id:
        path = suite_dir / f"{task_id}.json"
        if not path.is_file():
            _safe_echo(f"任务不存在：{path}", err=True)
            raise typer.Exit(1)
        tasks = [load_task(path)]
    else:
        tasks = load_tasks(suite_dir)

    if not tasks:
        _safe_echo(f"套件 {suite} 无任务（{suite_dir}）", err=True)
        raise typer.Exit(1)

    _safe_echo(f"开始评估：suite={suite} tasks={len(tasks)} trials={trials} threshold={quality_threshold:g} judge={judge}")

    runner = EvalRunner(quality_threshold=quality_threshold, enable_llm_judge=judge)

    async def _run_all():
        all_records = []
        for task in tasks:
            _safe_echo(f"▶ {task.task_id} {task.question[:30]}…")
            records = await runner.run_task(task, trials=trials)
            for record in records:
                status = "PASS" if record.pass_gates and record.quality >= quality_threshold else "FAIL"
                _safe_echo(
                    f"  trial {record.trial}: {status} quality={record.quality} "
                    f"gates={record.failed_gates or 'ok'} {record.latency_ms}ms"
                )
            all_records.extend(r.to_dict() for r in records)
        return all_records

    records = asyncio.run(_run_all())
    summary = summarize_trials(records, quality_threshold=quality_threshold)
    stability = pass_all_n(records, n_trials_per_task=trials, quality_threshold=quality_threshold)

    _safe_echo("\n==== 指标汇总 ====")
    _safe_echo(json.dumps(summary, ensure_ascii=False, indent=2))
    _safe_echo("\n==== Pass-all-N ====")
    _safe_echo(json.dumps(stability, ensure_ascii=False, indent=2))
    _safe_echo(f"\n报告目录：{runner.reports_root}")

if __name__ == "__main__":
    app()
