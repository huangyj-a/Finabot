"""Local memory system for Finabot."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


MEMORY_ROOT = Path(os.getenv("FINABOT_MEMORY_DIR", "memory"))
SHORT_TERM_DIR = MEMORY_ROOT / "short_term"
WORKING_MEMORY_DIR = MEMORY_ROOT / "working_memory"
LONG_TERM_DB = MEMORY_ROOT / "long_term.db"
KNOWLEDGE_DIR = MEMORY_ROOT / "knowledge"
LONG_TERM_KEYS = {"risk", "income", "goal", "taboo"}
KNOWLEDGE_COLLECTION = "finance_rules"


def _ensure_storage() -> None:
    for directory in (SHORT_TERM_DIR, WORKING_MEMORY_DIR, KNOWLEDGE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    _ensure_long_term_table()


def _safe_name(identifier: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier.strip())
    return name or "default"


def _json_path(directory: Path, identifier: str) -> Path:
    return directory / f"{_safe_name(identifier)}.json"


def _ensure_long_term_table() -> None:
    try:
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(LONG_TERM_DB) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memory_user_key
                ON user_memory(user_id, key)
                """
            )
    except Exception:
        return


def save_short_memory(session_id: str, messages: list) -> None:
    """Persist session chat history as UTF-8 JSON."""

    try:
        _ensure_storage()
        path = _json_path(SHORT_TERM_DIR, session_id)
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def load_short_memory(session_id: str) -> list:
    """Load session chat history; return [] when missing or invalid."""

    try:
        _ensure_storage()
        path = _json_path(SHORT_TERM_DIR, session_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_working_memory(task_id: str, data: dict) -> None:
    """Persist agent runtime state as UTF-8 JSON."""

    try:
        _ensure_storage()
        path = _json_path(WORKING_MEMORY_DIR, task_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def load_working_memory(task_id: str) -> dict:
    """Load agent runtime state; return {} when missing or invalid."""

    try:
        _ensure_storage()
        path = _json_path(WORKING_MEMORY_DIR, task_id)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_long_term(user_id: str, key: str, value: str) -> None:
    """Save permanent user memory for supported keys: risk, income, goal, taboo."""

    normalized_key = key.strip().lower()
    if normalized_key not in LONG_TERM_KEYS:
        return
    try:
        _ensure_storage()
        with sqlite3.connect(LONG_TERM_DB) as conn:
            conn.execute(
                """
                INSERT INTO user_memory(user_id, key, value, update_time)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user_id, normalized_key, value),
            )
    except Exception:
        return


def get_long_term(user_id: str, key: str) -> str | None:
    """Return latest value for a user's memory key."""

    try:
        _ensure_storage()
        with sqlite3.connect(LONG_TERM_DB) as conn:
            row = conn.execute(
                """
                SELECT value FROM user_memory
                WHERE user_id = ? AND key = ?
                ORDER BY update_time DESC, id DESC
                LIMIT 1
                """,
                (user_id, key.strip().lower()),
            ).fetchone()
        return str(row[0]) if row else None
    except Exception:
        return None


def get_all_user_memory(user_id: str) -> dict:
    """Return latest permanent memory values for the user."""

    memories: dict[str, str] = {}
    for key in LONG_TERM_KEYS:
        value = get_long_term(user_id, key)
        if value is not None:
            memories[key] = value
    return memories


def add_knowledge(doc_id: str | int, content: str) -> None:
    """Add or update local finance knowledge in Chroma when available."""

    try:
        collection = _get_knowledge_collection()
        if collection is None:
            _fallback_add_knowledge(str(doc_id), content)
            return
        collection.upsert(ids=[str(doc_id)], documents=[content])
    except Exception:
        _fallback_add_knowledge(str(doc_id), content)


def query_knowledge(question: str, n_results: int = 2) -> list[str]:
    """Query related local finance knowledge."""

    try:
        collection = _get_knowledge_collection()
        if collection is None:
            return _fallback_query_knowledge(question, n_results)
        result = collection.query(query_texts=[question], n_results=n_results)
        documents = result.get("documents", [[]])
        return [str(item) for item in documents[0]] if documents else []
    except Exception:
        return _fallback_query_knowledge(question, n_results)


def build_memory_context(session_id: str, user_id: str, question: str, max_short_messages: int = 8) -> list[dict[str, Any]]:
    """Collect short-term, long-term, and knowledge memories for prompt building."""

    memories: list[dict[str, Any]] = []
    short_memory = load_short_memory(session_id)[-max_short_messages:]
    if short_memory:
        memories.append({"summary": "短期对话记忆", "content": _format_short_memory(short_memory)})

    user_memory = get_all_user_memory(user_id)
    if user_memory:
        memories.append({"summary": "用户长期画像", "content": _format_long_memory(user_memory)})

    knowledge = query_knowledge(question, n_results=2)
    if knowledge:
        memories.append({"summary": "相关金融知识", "content": "\n".join(f"- {item}" for item in knowledge)})
    return memories


def _format_short_memory(messages: list) -> str:
    lines = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "unknown")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "unknown")
            content = getattr(message, "content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_long_memory(memories: dict[str, str]) -> str:
    labels = {"risk": "风险偏好", "income": "收入/资金情况", "goal": "投资目标", "taboo": "禁忌/约束"}
    return "\n".join(f"- {labels.get(key, key)}：{value}" for key, value in sorted(memories.items()))


def _get_knowledge_collection():
    try:
        _ensure_storage()
        import chromadb

        client = chromadb.PersistentClient(path=str(KNOWLEDGE_DIR))
        return client.get_or_create_collection(KNOWLEDGE_COLLECTION)
    except Exception:
        return None


def _fallback_knowledge_path() -> Path:
    return KNOWLEDGE_DIR / "finance_rules.json"


def _fallback_add_knowledge(doc_id: str, content: str) -> None:
    try:
        _ensure_storage()
        path = _fallback_knowledge_path()
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[str(doc_id)] = content
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _fallback_query_knowledge(question: str, n_results: int) -> list[str]:
    try:
        path = _fallback_knowledge_path()
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        terms = {term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", question)}
        scored = []
        for content in data.values():
            text = str(content)
            score = sum(1 for term in terms if term and term in text.lower())
            scored.append((score, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for score, text in scored[:n_results] if score > 0]
    except Exception:
        return []


_ensure_storage()
