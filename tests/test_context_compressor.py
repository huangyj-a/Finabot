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
            max_messages=9,
            keep_head_messages=2,
            keep_tail_messages=5,
            keep_recent_tool_results=1,
            spill_dir=temp_dir / "spill",
        )
    )
    try:
        messages = [{"role": "system", "content": "system"}]
        messages.extend({"role": "user", "content": f"message {index}"} for index in range(6))
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tool-1", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "A" * 500, "tool_call_id": "tool-1"},
                {"role": "assistant", "content": "after tool"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "tool-2", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
                    ],
                },
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


def _internal_tool_call_pair_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "result-1", "tool_call_id": "t1"},
        {"role": "assistant", "content": "final answer"},
    ]


def test_context_compressor_drops_orphan_tool_result_after_snip():
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(
            max_messages=4,
            keep_head_messages=1,
            keep_tail_messages=2,
            spill_dir=temp_dir / "spill",
        )
    )
    try:
        compressed = compressor.compress(_internal_tool_call_pair_messages())

        roles = [message["role"] for message in compressed]
        # t1 的 assistant 消息被裁掉后，孤儿 tool 结果必须被丢弃
        assert "tool" not in roles
        dangling = [
            message for message in compressed
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        assert dangling == []
        assert compressed[-1]["content"] == "final answer"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_context_compressor_reactive_repairs_split_pair():
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(emergency_keep_last_messages=2, spill_dir=temp_dir / "spill")
    )
    try:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t9", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "orphan result", "tool_call_id": "t9"},
            {"role": "assistant", "content": "done"},
        ]

        compressed = compressor.compress(messages, mode="reactive")

        assert all(message.get("role") != "tool" for message in compressed)
        assert compressed[-1]["content"] == "done"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_context_compressor_keeps_intact_tool_pair_in_recent_window():
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(emergency_keep_last_messages=3, spill_dir=temp_dir / "spill")
    )
    try:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "keep-1", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "kept result", "tool_call_id": "keep-1"},
            {"role": "assistant", "content": "done"},
        ]

        compressed = compressor.compress(messages, mode="reactive")

        tool_messages = [message for message in compressed if message.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "keep-1"
        # 配对完整保留：tool 消息前一条必须是带对应 tool_calls 的 assistant
        previous = compressed[compressed.index(tool_messages[0]) - 1]
        assert previous["role"] == "assistant"
        assert any(
            call.get("id") == "keep-1" for call in previous.get("tool_calls", [])
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cjk_auto_compact_fires_with_chinese_token_weighting():
    """中文每字≈1 token：旧 chars//4 估算会把 57 字算成 14 token，低于阈值不触发；
    CJK 加权估算为 ~57 token，应触发 auto 压缩。"""
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
        messages.extend({"role": "user", "content": "测试压缩"} for _ in range(14))

        compressed = compressor.compress(messages)
        contents = "\n".join(str(message.get("content", "")) for message in compressed)

        assert "[autoCompact/sessionMemoryCompact]" in contents
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_micro_compact_spills_old_tool_result_with_read_file_pointer():
    """旧工具结果不再只留占位：落盘并给出 read_file 指针，内容可恢复。"""
    temp_dir = _internal_repo_temp_dir()
    compressor = ContextCompressor(
        ContextCompressionConfig(keep_recent_tool_results=1, spill_dir=temp_dir / "spill")
    )
    try:
        messages = [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "old data " + "x" * 120, "tool_call_id": "t1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t2", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "recent result", "tool_call_id": "t2"},
        ]

        compressed = compressor.compress(messages)
        old_tool = next(
            message for message in compressed
            if message.get("role") == "tool" and message.get("tool_call_id") == "t1"
        )
        recent_tool = next(
            message for message in compressed
            if message.get("role") == "tool" and message.get("tool_call_id") == "t2"
        )

        assert "[microCompact]" in old_tool["content"]
        assert "read_file" in old_tool["content"]
        assert recent_tool["content"] == "recent result"
        assert (temp_dir / "spill").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
