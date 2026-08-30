"""Tests for the eval task schema and loader."""

from finabot.eval.tasks import EvalTask, find_task_root, load_task, load_task_by_id, load_tasks


def test_find_task_root_points_to_eval_tasks():
    root = find_task_root()
    assert root.name == "tasks"
    assert root.parent.name == "eval"


def test_load_task_parses_all_fields():
    root = find_task_root()
    path = root / "dev" / "t001_timing_leak.json"
    assert path.is_file(), "t001 task file must exist"
    task = load_task(path)
    assert task.task_id == "t001"
    assert task.suite == "dev"
    assert task.as_of  # as_of 为采样数据日期，非空即可（具体值随快照更新）
    assert task.question
    assert task.hard_gates
    assert task.graders
    assert task.budget["max_llm_calls"] > 0


def test_load_tasks_dev_returns_twenty():
    root = find_task_root()
    tasks = load_tasks(root / "dev")
    assert len(tasks) >= 20, f"dev suite should have >=20 tasks, got {len(tasks)}"
    ids = [t.task_id for t in tasks]
    assert ids == sorted(ids)
    assert "t001" in ids and "t020" in ids


def test_load_task_by_id():
    task = load_task_by_id("t007")
    assert task is not None
    assert task.task_id == "t007"
    assert "买入" in task.question


def test_from_dict_defaults():
    task = EvalTask.from_dict({"task_id": "x1"})
    assert task.suite == "dev"
    assert task.budget["max_llm_calls"] == 8
    assert task.allowed_sources == ["cninfo", "sse", "eastmoney"]