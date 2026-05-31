import importlib
import os
import shutil
from pathlib import Path
from uuid import uuid4


def _internal_repo_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / ".test_tmp"
    path = root / uuid4().hex
    path.mkdir(parents=True)
    return path


def _load_memory_module(temp_dir: Path):
    os.environ["FINABOT_MEMORY_DIR"] = str(temp_dir / "memory")
    import finabot.agents.memory as memory

    return importlib.reload(memory)


def test_short_and_working_memory_roundtrip():
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)

        memory.save_short_memory("cli:demo", [{"role": "user", "content": "你好"}])
        assert memory.load_short_memory("cli:demo") == [{"role": "user", "content": "你好"}]

        memory.save_working_memory("task:1", {"step": 1, "result": "ok"})
        assert memory.load_working_memory("task:1") == {"step": 1, "result": "ok"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)


def test_long_term_memory_keeps_latest_supported_key():
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)

        memory.save_long_term("user-1", "risk", "稳健")
        memory.save_long_term("user-1", "risk", "中等风险")
        memory.save_long_term("user-1", "unknown", "ignored")

        assert memory.get_long_term("user-1", "risk") == "中等风险"
        assert memory.get_all_user_memory("user-1") == {"risk": "中等风险"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)


def test_knowledge_fallback_and_memory_context(monkeypatch):
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)
        monkeypatch.setattr(memory, "_get_knowledge_collection", lambda: None)

        memory.save_short_memory("session-1", [{"role": "user", "content": "我偏稳健"}])
        memory.save_long_term("user-1", "goal", "长期增值")
        memory.add_knowledge("rule-1", "稳健投资应关注最大回撤和仓位控制")

        assert memory.query_knowledge("稳健投资", n_results=1) == ["稳健投资应关注最大回撤和仓位控制"]
        context = memory.build_memory_context("session-1", "user-1", "稳健投资")
        content = "\n".join(item["content"] for item in context)

        assert "我偏稳健" in content
        assert "长期增值" in content
        assert "最大回撤" in content
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)
