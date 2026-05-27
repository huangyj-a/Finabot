from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class SessionManager:
    def __init__(self, ttl_minutes: int = 60):
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.last_access: Dict[str, datetime] = {}

    def _cleanup_expired(self):
        """清理过期会话"""
        now = datetime.now()
        expired_keys = [
            key for key, last in self.last_access.items()
            if now - last > self.ttl
        ]
        for key in expired_keys:
            self.sessions.pop(key, None)
            self.last_access.pop(key, None)

    def add_message(self, session_key: str, role: str, content: str):
        self._cleanup_expired()
        if session_key not in self.sessions:
            self.sessions[session_key] = []
        self.sessions[session_key].append(
            ChatMessage(role=role, content=content, timestamp=datetime.now())
        )
        self.last_access[session_key] = datetime.now()

    def get_messages(self, session_key: str) -> List[dict]:
        self._cleanup_expired()
        history = self.sessions.get(session_key, [])
        return [{"role": m.role, "content": m.content} for m in history]