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


def test_extract_user_profile_heuristics():
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)

        profile = memory.extract_user_profile(
            "我是稳健型投资者，偏好长期价值投资，不碰科技股，资金约50万"
        )
        assert profile.get("risk") == "稳健"
        assert profile.get("goal") == "长期增值"
        assert profile.get("taboo") == "不碰科技股"
        assert profile.get("income") == "约50万"

        # 无信号时不臆测
        assert memory.extract_user_profile("茅台今天怎么样") == {}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)


def test_record_stock_dedupes_and_caps():
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)

        memory.record_stock("user-1", "600519", "贵州茅台")
        memory.record_stock("user-1", "300502", "新易盛")
        memory.record_stock("user-1", "600519", "贵州茅台")  # 去重，仍置顶

        stocks = memory.get_long_term("user-1", "stocks")
        assert stocks is not None
        assert "600519" in stocks
        assert "贵州茅台" in stocks
        # 去重后应只有两条
        items = memory._load_json_list("user-1", "stocks")
        assert len(items) == 2
        assert items[0]["code"] == "600519"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)


def test_record_conclusion_prepends():
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)

        memory.record_conclusion("user-1", "贵州茅台", "结论A")
        memory.record_conclusion("user-1", "贵州茅台", "结论B")

        items = memory._load_json_list("user-1", "conclusions")
        assert len(items) == 2
        assert items[0]["conclusion"] == "结论B"
        assert items[1]["conclusion"] == "结论A"
        assert items[0]["stock"] == "贵州茅台"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)


def test_record_run_memory_wires_everything(monkeypatch):
    temp_dir = _internal_repo_temp_dir()
    try:
        memory = _load_memory_module(temp_dir)
        monkeypatch.setattr(memory, "_get_knowledge_collection", lambda: None)

        final_state = {
            "akshare_cache": {
                "600519": {
                    "resolved_symbol": "600519",
                    "resolved_name": "贵州茅台",
                    "stock_spot": "{}",
                }
            }
        }
        memory.record_run_memory(
            "user-1",
            "我是稳健型投资者，贵州茅台适合持有吗",
            final_state,
            "建议持有，长期看好",
        )

        # 画像
        assert memory.get_long_term("user-1", "risk") == "稳健"
        # 关注股票
        stocks = memory._load_json_list("user-1", "stocks")
        assert stocks[0]["code"] == "600519"
        assert stocks[0]["name"] == "贵州茅台"
        # 历史结论
        conclusions = memory._load_json_list("user-1", "conclusions")
        assert conclusions[0]["stock"] == "贵州茅台"
        assert conclusions[0]["conclusion"] == "建议持有，长期看好"

        # 注入到下一轮上下文（新对话自动带上记忆）
        context = memory.build_memory_context("session-2", "user-1", "贵州茅台现在还能拿吗")
        content = "\n".join(item["content"] for item in context)
        assert "稳健" in content
        assert "关注股票" in content
        assert "贵州茅台" in content
        assert "历史分析结论" in content
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop("FINABOT_MEMORY_DIR", None)
