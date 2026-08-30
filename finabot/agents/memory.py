"""Local memory system for Finabot."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

MEMORY_ROOT = Path(os.getenv("FINABOT_MEMORY_DIR", "memory"))
SHORT_TERM_DIR = MEMORY_ROOT / "short_term"
WORKING_MEMORY_DIR = MEMORY_ROOT / "working_memory"
LONG_TERM_DB = MEMORY_ROOT / "long_term.db"
KNOWLEDGE_DIR = MEMORY_ROOT / "knowledge"
LONG_TERM_KEYS = {"risk", "income", "goal", "taboo", "stocks", "conclusions"}
KNOWLEDGE_COLLECTION = "finance_rules"

# 关注股票 / 历史结论的沉淀上限（防止长期记忆无限膨胀挤占提示词）
_STOCKS_CAP = 10
_CONCLUSIONS_CAP = 10
_CONCLUSION_MAX_CHARS = 200


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
    except Exception as exc:
        logger.warning("init long_term table failed; user memory disabled: %s", exc)


def save_short_memory(session_id: str, messages: list) -> None:
    """Persist session chat history as UTF-8 JSON."""

    try:
        _ensure_storage()
        path = _json_path(SHORT_TERM_DIR, session_id)
        path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        # 写入失败意味着聊天记录丢失，必须留下可观测的痕迹
        logger.warning("save_short_memory failed for %s: %s", session_id, exc)


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
    except Exception as exc:
        logger.warning("save_working_memory failed for %s: %s", task_id, exc)


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


# ---------------------------------------------------------------------------
# 记忆沉淀（生产写入路径）
# 说明：risk/income/goal/taboo 是单值画像；stocks（关注股票）与 conclusions
# （历史分析结论）是 JSON 数组，采用"读最新-合并-回写"的方式累积，带上限。
# ---------------------------------------------------------------------------

# 用户画像关键词规则：(key, [(关键词, 归一化值), ...])，命中即取首个
_PROFILE_KEYWORD_RULES: list[tuple[str, list[tuple[str, str]]]] = [
    ("risk", [("稳健", "稳健"), ("保守", "保守"), ("平衡", "中等风险"), ("激进", "激进"), ("高收益", "激进")]),
    ("goal", [("长期", "长期增值"), ("价值投资", "长期增值"), ("短线", "短期交易"), ("短期", "短期交易")]),
]


def extract_user_profile(message: str) -> dict[str, str]:
    """从用户消息中启发式提取画像信号（无信号则不返回该 key）。

    仅做确定性关键词/正则匹配，不消耗 LLM；识别不到的偏好保持沉默，
    避免把猜测写进长期记忆。
    """
    text = str(message or "")
    profile: dict[str, str] = {}

    for key, rules in _PROFILE_KEYWORD_RULES:
        for keyword, value in rules:
            if keyword in text:
                profile[key] = value
                break

    taboo_match = re.search(r"(?:不|别|避免|禁)[^，。！？!?；;]{0,12}", text)
    if taboo_match and taboo_match.group(0).strip():
        profile["taboo"] = taboo_match.group(0).strip()

    income_match = re.search(r"(?:资金|本金|投入|规模)[约\s]*([0-9一二三四五六七八九十百千万.]+万?)", text)
    if income_match:
        profile["income"] = f"约{income_match.group(1)}"

    return profile


def _load_json_list(user_id: str, key: str) -> list:
    raw = get_long_term(user_id, key)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []


def _save_json_list(user_id: str, key: str, items: list) -> None:
    save_long_term(user_id, key, json.dumps(items, ensure_ascii=False))


def record_stock(user_id: str, code: str, name: str = "") -> None:
    """沉淀一只关注/近期分析的股票（按代码去重，新进置顶，带上限）。"""
    code = str(code or "").strip()
    if not code:
        return
    items = _load_json_list(user_id, "stocks")
    items = [item for item in items if not (isinstance(item, dict) and item.get("code") == code)]
    items.insert(0, {"code": code, "name": str(name or "").strip()})
    _save_json_list(user_id, "stocks", items[:_STOCKS_CAP])


def record_conclusion(user_id: str, stock: str, conclusion: str) -> None:
    """沉淀一条历史分析结论（新进置顶，带上限与截断）。"""
    from datetime import datetime

    items = _load_json_list(user_id, "conclusions")
    items.insert(
        0,
        {
            "stock": str(stock or "")[:40],
            "date": datetime.now().date().isoformat(),
            "conclusion": str(conclusion or "")[:_CONCLUSION_MAX_CHARS],
        },
    )
    _save_json_list(user_id, "conclusions", items[:_CONCLUSIONS_CAP])


def record_run_memory(
    user_id: str,
    question: str,
    final_state: dict | None,
    reply: str,
) -> None:
    """一轮对话结束后沉淀长期记忆：用户画像、关注股票、历史分析结论。

    从最终状态的 AKShare 缓存读取本轮已解析的股票代码/名称（比从提问文本
    猜名字可靠）；所有写入均为尽力而为，异常不阻断主流程。
    """
    try:
        profile = extract_user_profile(question)
        for key, value in profile.items():
            save_long_term(user_id, key, value)

        cache = (final_state or {}).get("akshare_cache") or {}
        seen: set[str] = set()
        for payload in cache.values():
            if not isinstance(payload, dict):
                continue
            code = str(payload.get("resolved_symbol") or "").strip()
            if not code or code in seen:
                continue
            seen.add(code)
            record_stock(user_id, code, str(payload.get("resolved_name") or "").strip())

        stock_label = ""
        for payload in cache.values():
            if isinstance(payload, dict) and (payload.get("resolved_name") or payload.get("resolved_symbol")):
                stock_label = str(payload.get("resolved_name") or payload.get("resolved_symbol") or "")
                break
        # 只有本轮确实分析了股票、或答复是实质性内容时才沉淀结论，避免"你好"
        # 这类寒暄对话污染历史结论。
        if stock_label or len(str(reply or "")) >= 20:
            record_conclusion(user_id, stock_label or question, reply)
    except Exception as exc:
        logger.warning("record_run_memory failed for user %s: %s", user_id, exc)


def add_knowledge(doc_id: str | int, content: str) -> None:
    """Add or update local finance knowledge in Chroma when available."""

    try:
        collection = _get_knowledge_collection()
        if collection is None:
            _fallback_add_knowledge(str(doc_id), content)
            return
        collection.upsert(ids=[str(doc_id)], documents=[content])
    except Exception as exc:
        logger.warning("add_knowledge upsert failed for %s: %s", doc_id, exc)
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
    labels = {
        "risk": "风险偏好",
        "income": "收入/资金情况",
        "goal": "投资目标",
        "taboo": "禁忌/约束",
        "stocks": "关注股票",
        "conclusions": "历史分析结论",
    }
    lines: list[str] = []
    for key, value in sorted(memories.items()):
        label = labels.get(key, key)
        if key == "stocks":
            try:
                stocks = json.loads(value)
            except (ValueError, TypeError):
                stocks = []
            names = []
            for item in stocks[:5]:
                if isinstance(item, dict):
                    names.append(item.get("name") or item.get("code") or "")
            if names:
                lines.append(f"- {label}：" + "、".join(names))
        elif key == "conclusions":
            try:
                conclusions = json.loads(value)
            except (ValueError, TypeError):
                conclusions = []
            for item in conclusions[:3]:
                if isinstance(item, dict):
                    lines.append(
                        f"- {label}（{item.get('date', '')}，{item.get('stock', '')}）：{item.get('conclusion', '')}"
                    )
        else:
            lines.append(f"- {label}：{value}")
    return "\n".join(lines)


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


try:
    _ensure_storage()
except Exception as exc:  # 导入期副作用：存储初始化失败不应让 import 崩溃
    logger.warning("memory storage init failed at import: %s", exc)
