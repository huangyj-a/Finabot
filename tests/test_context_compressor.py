import shutil
from pathlib import Path
from uuid import uuid4

from finabot.agents.context import ContextCompressionConfig, ContextCompressor


def _internal_repo_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / ".test_tmp"
    path = root / uuid4().hex
    path.mkdir(parents=True)
    return path


def test_context_compressor_runs_l3_l1_l2_pipeline():
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(
            tool_result_budget_bytes=160,
            max_messages=8,
            keep_head_messages=2,
            keep_tail_messages=4,
            keep_recent_tool_results=1,
            spill_dir=temp_dir / "spill",
        )
    )
    try:
        messages = [{"role": "system", "content": "system"}]
        messages.extend({"role": "user", "content": f"message {index}"} for index in range(6))
        messages.extend(
            [
                {"role": "tool", "content": "A" * 500, "tool_call_id": "tool-1"},
                {"role": "assistant", "content": "after tool"},
                {"role": "tool", "content": "recent tool result", "tool_call_id": "tool-2"},
            ]
        )

        compressed = compressor.compress(messages)
        contents = "\n".join(str(message.get("content", "")) for message in compressed)

        assert "[toolResultBudget]" in contents
        assert "[snipCompact]" in contents
        assert "recent tool result" in contents
        assert (temp_dir / "spill").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_context_compressor_auto_compacts_when_over_threshold():
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(
            context_window_tokens=40,
            max_output_tokens=5,
            auto_compact_margin_tokens=5,
            spill_dir=temp_dir / "spill",
        )
    )
    try:
        messages = [{"role": "system", "content": "system"}]
        messages.extend({"role": "user", "content": "long message " * 20} for _ in range(14))

        compressed = compressor.compress(messages)
        contents = "\n".join(str(message.get("content", "")) for message in compressed)

        assert "[autoCompact/sessionMemoryCompact]" in contents
        assert len(compressed) < len(messages)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_context_compressor_reactive_keeps_recent_messages():
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(emergency_keep_last_messages=3, spill_dir=temp_dir / "spill")
    )
    try:
        messages = [{"role": "system", "content": "system"}]
        messages.extend({"role": "user", "content": f"old {index}"} for index in range(8))
        messages.extend({"role": "assistant", "content": f"recent {index}"} for index in range(3))

        compressed = compressor.compress(messages, mode="reactive")
        contents = "\n".join(str(message.get("content", "")) for message in compressed)

        assert "[reactiveCompact]" in contents
        assert "recent 0" in contents
        assert "recent 1" in contents
        assert "recent 2" in contents
        assert "old 0" in contents
        assert len(compressed) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
