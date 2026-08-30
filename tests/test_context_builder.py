import shutil
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from finabot.agents.context import ContextBuilder


def _internal_repo_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / ".test_tmp"
    path = root / uuid4().hex
    path.mkdir(parents=True)
    return path


def test_context_builder_loads_always_skills_and_summarizes_on_demand():
    temp_dir = _internal_repo_temp_dir()
    try:
        skills_root = temp_dir / "skills"
        skills_root.mkdir()
        (skills_root / "always.md").write_text(
            """---
name: Always Skill
summary: always summary
always: true
---
# Always Body
始终加载的完整技能内容。
""",
            encoding="utf-8",
        )
        (skills_root / "on_demand.md").write_text(
            """---
name: On Demand Skill
summary: 按需技能摘要
always: false
---
# On Demand Body
按需技能的完整正文不应直接进入系统提示。
""",
            encoding="utf-8",
        )

        builder = ContextBuilder("基础系统提示", skills_root=skills_root)
        prompt = builder.build_system_prompt(memories=[{"content": "历史记忆 A"}])

        assert "基础系统提示" in prompt
        assert "历史记忆 A" in prompt
        assert "始终加载的完整技能内容" in prompt
        assert "按需技能摘要" in prompt
        assert "按需技能的完整正文" not in prompt
        assert "read_file(path)" in prompt
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_context_builder_preserves_system_messages_and_tool_calls():
    temp_dir = _internal_repo_temp_dir()
    builder = ContextBuilder("基础系统提示", skills_root=temp_dir / "missing")
    messages = [
        SystemMessage(content="子代理系统提示"),
        HumanMessage(content="用户问题"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "skills/demo.md"}, "id": "call-1"}],
        ),
        ToolMessage(content="demo 内容", tool_call_id="call-1"),
    ]

    converted = builder.build_messages(messages)

    assert converted[0]["role"] == "system"
    assert "基础系统提示" in converted[0]["content"]
    assert "子代理系统提示" in converted[0]["content"]
    assert converted[1] == {"role": "user", "content": "用户问题"}
    assert converted[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert converted[3]["tool_call_id"] == "call-1"
    shutil.rmtree(temp_dir, ignore_errors=True)
