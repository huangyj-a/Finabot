"""Context building and compression for LLM prompts."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from langchain_core.messages import BaseMessage


CompressionMode = Literal["auto", "reactive", "off"]


@dataclass(frozen=True)
class ContextCompressionConfig:
    tool_result_budget_bytes: int = 200 * 1024
    max_messages: int = 50
    keep_head_messages: int = 10
    keep_tail_messages: int = 40
    keep_recent_tool_results: int = 3
    context_window_tokens: int = 128_000
    max_output_tokens: int = 4_096
    auto_compact_margin_tokens: int = 13_000
    emergency_keep_last_messages: int = 5
    chars_per_token: int = 4
    spill_dir: Path = Path(".finabot_context/tool_results")

    @classmethod
    def from_env(cls) -> "ContextCompressionConfig":
        return cls(
            context_window_tokens=int(os.getenv("FINABOT_CONTEXT_WINDOW", "128000")),
            max_output_tokens=int(os.getenv("FINABOT_MAX_OUTPUT_TOKENS", "4096")),
            spill_dir=Path(os.getenv("FINABOT_CONTEXT_CACHE_DIR", ".finabot_context/tool_results")),
        )


class ContextCompressor:
    """Apply L3 → L1 → L2 preprocessing, auto compact, and reactive fallback."""

    def __init__(self, config: ContextCompressionConfig | None = None):
        self.config = config or ContextCompressionConfig.from_env()

    def compress(self, messages: list[dict], mode: CompressionMode = "auto") -> list[dict]:
        if mode == "off":
            return [dict(message) for message in messages]

        compressed = [dict(message) for message in messages]
        compressed = self._tool_result_budget(compressed)
        compressed = self._snip_compact(compressed)
        compressed = self._micro_compact(compressed)

        if mode == "reactive":
            return self._reactive_compact(compressed)
        if self._estimated_tokens(compressed) > self._auto_compact_threshold():
            compressed = self._session_memory_compact(compressed)
        return compressed

    def _tool_result_budget(self, messages: list[dict]) -> list[dict]:
        tool_indices = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
        total_bytes = sum(self._content_bytes(messages[index]) for index in tool_indices)
        if total_bytes <= self.config.tool_result_budget_bytes:
            return messages

        largest_index = max(tool_indices, key=lambda index: self._content_bytes(messages[index]))
        content = str(messages[largest_index].get("content", ""))
        if not content or content.startswith("[toolResultBudget]"):
            return messages
        path = self._spill_content(content)
        display_path = self._display_path(path)
        original_bytes = len(content.encode("utf-8"))
        messages[largest_index]["content"] = (
            f"[toolResultBudget] 工具结果已落盘保留完整内容；"
            f"原始大小 {original_bytes} bytes；如需完整内容，调用 read_file 读取 `{display_path}`。"
        )
        return messages

    def _snip_compact(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= self.config.max_messages:
            return messages

        system_messages = [message for message in messages if message.get("role") == "system"]
        non_system = [message for message in messages if message.get("role") != "system"]
        head = non_system[: self.config.keep_head_messages]
        tail = non_system[-self.config.keep_tail_messages :]
        omitted = max(len(non_system) - len(head) - len(tail), 0)
        marker = {
            "role": "system",
            "content": f"[snipCompact] 已裁剪中间 {omitted} 条旧消息，仅保留开头和最近上下文。",
        }
        return system_messages[:1] + [marker] + head + tail

    def _micro_compact(self, messages: list[dict]) -> list[dict]:
        tool_indices = [index for index, message in enumerate(messages) if message.get("role") == "tool"]
        compact_indices = set(tool_indices[: -self.config.keep_recent_tool_results])
        for index in compact_indices:
            content = str(messages[index].get("content", ""))
            if content.startswith("[toolResultBudget]"):
                continue
            messages[index]["content"] = (
                f"[microCompact] 旧工具结果已压缩，占位保留。"
                f"原始字符数 {len(content)}；最近 {self.config.keep_recent_tool_results} 条工具结果保留全文。"
            )
        return messages

    def _session_memory_compact(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 12:
            return messages
        system = messages[:1]
        recent = messages[-10:]
        middle = messages[1:-10]
        summary = "\n".join(self._summarize_message(message) for message in middle if message.get("content"))
        summary_message = {
            "role": "system",
            "content": "[autoCompact/sessionMemoryCompact] 早期上下文摘要：\n" + summary,
        }
        return system + [summary_message] + recent

    def _reactive_compact(self, messages: list[dict]) -> list[dict]:
        system = messages[:1]
        recent = messages[-self.config.emergency_keep_last_messages :]
        older = messages[1 : -self.config.emergency_keep_last_messages]
        summary = "\n".join(self._summarize_message(message, max_chars=160) for message in older if message.get("content"))
        emergency = {
            "role": "system",
            "content": "[reactiveCompact] prompt_too_long 应急压缩摘要：\n" + summary,
        }
        return system + [emergency] + recent

    def _spill_content(self, content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        self.config.spill_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.spill_dir / f"tool_result_{digest}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()

    def _content_bytes(self, message: dict) -> int:
        return len(str(message.get("content", "")).encode("utf-8"))

    def _estimated_tokens(self, messages: list[dict]) -> int:
        chars = sum(len(str(message.get("content", ""))) for message in messages)
        return max(chars // max(self.config.chars_per_token, 1), 1)

    def _auto_compact_threshold(self) -> int:
        return max(
            self.config.context_window_tokens
            - self.config.max_output_tokens
            - self.config.auto_compact_margin_tokens,
            1,
        )

    def _summarize_message(self, message: dict, max_chars: int = 240) -> str:
        content = str(message.get("content", "")).replace("\n", " ").strip()
        if len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}..."
        return f"- {message.get('role', 'unknown')}: {content}"
@dataclass(frozen=True)
class SkillDescriptor:
    name: str
    path: str
    summary: str
    always: bool
    content: str


def _internal_parse_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    if not raw_text.startswith("---"):
        return {}, raw_text.strip()

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text.strip()

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip().strip('"\'')
    return metadata, parts[2].strip()


def _internal_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _internal_summary_from_body(body: str, max_chars: int = 240) -> str:
    paragraphs = [line.strip("# ").strip() for line in body.splitlines() if line.strip()]
    if not paragraphs:
        return "无摘要。"
    summary = paragraphs[0]
    return summary if len(summary) <= max_chars else f"{summary[:max_chars].rstrip()}..."


class ContextBuilder:
    """Assemble system prompt, memories, skills, and conversation history."""

    def __init__(
        self,
        base_system_prompt: str,
        skills_root: str | os.PathLike[str] | None = None,
        compression_config: ContextCompressionConfig | None = None,
    ):
        self.base_system_prompt = base_system_prompt.strip()
        root = skills_root or os.getenv("FINABOT_SKILLS_DIR") or "skills"
        self.skills_root = Path(root)
        self.compressor = ContextCompressor(compression_config)

    def discover_skills(self) -> list[SkillDescriptor]:
        if not self.skills_root.exists():
            return []

        skills: list[SkillDescriptor] = []
        for path in sorted(self.skills_root.rglob("*.md")):
            if not path.is_file():
                continue
            raw_text = path.read_text(encoding="utf-8")
            metadata, body = _internal_parse_frontmatter(raw_text)
            relative_path = path.as_posix()
            name = metadata.get("name") or path.stem
            summary = metadata.get("summary") or metadata.get("description") or _internal_summary_from_body(body)
            skills.append(
                SkillDescriptor(
                    name=name,
                    path=relative_path,
                    summary=summary,
                    always=_internal_bool(metadata.get("always")),
                    content=body,
                )
            )
        return skills

    def build_system_prompt(
        self,
        memories: Iterable[str | dict[str, Any]] | None = None,
        extra_system_prompts: Sequence[str] | None = None,
    ) -> str:
        sections = [self.base_system_prompt]
        if extra_system_prompts:
            sections.append("## 当前子代理系统提示\n" + "\n\n".join(prompt.strip() for prompt in extra_system_prompts if prompt.strip()))

        memory_section = self._format_memories(memories)
        if memory_section:
            sections.append(memory_section)

        skill_section = self._format_skills(self.discover_skills())
        if skill_section:
            sections.append(skill_section)

        return "\n\n".join(section for section in sections if section.strip())

    def build_messages(
        self,
        messages: list[BaseMessage],
        memories: Iterable[str | dict[str, Any]] | None = None,
        compression_mode: CompressionMode = "auto",
    ) -> list[dict]:
        system_prompts = [str(msg.content) for msg in messages if getattr(msg, "type", None) == "system"]
        converted = [{"role": "system", "content": self.build_system_prompt(memories, system_prompts)}]

        for msg in messages:
            message_type = getattr(msg, "type", None)
            if message_type == "system":
                continue
            if message_type == "human":
                converted.append({"role": "user", "content": msg.content})
            elif message_type == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": getattr(msg, "tool_call_id", None),
                    }
                )
            elif message_type == "ai":
                converted.append(self._convert_ai_message(msg))
        return self.compressor.compress(converted, mode=compression_mode)

    def _format_memories(self, memories: Iterable[str | dict[str, Any]] | None) -> str:
        if not memories:
            return ""
        lines = []
        for index, memory in enumerate(memories, 1):
            if isinstance(memory, dict):
                content = memory.get("content") or memory.get("recommendation") or memory.get("summary") or str(memory)
            else:
                content = str(memory)
            if content.strip():
                lines.append(f"{index}. {content.strip()}")
        if not lines:
            return ""
        return "## 记忆\n" + "\n".join(lines)

    def _format_skills(self, skills: list[SkillDescriptor]) -> str:
        if not skills:
            return ""

        always_skills = [skill for skill in skills if skill.always]
        on_demand_skills = [skill for skill in skills if not skill.always]
        lines = ["## 技能", "你可以使用 `read_file(path)` 按需读取技能完整内容。"]

        if always_skills:
            lines.append("### 始终加载")
            for skill in always_skills:
                lines.append(f"#### {skill.name}\n路径：`{skill.path}`\n{skill.content}")

        if on_demand_skills:
            lines.append("### 按需加载")
            for skill in on_demand_skills:
                lines.append(f"- {skill.name}：{skill.summary}（路径：`{skill.path}`）")
        return "\n".join(lines)

    def _convert_ai_message(self, msg: BaseMessage) -> dict:
        payload = {"role": "assistant", "content": getattr(msg, "content", "")}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            payload["tool_calls"] = [self._convert_tool_call(call) for call in tool_calls]
        return payload

    def _convert_tool_call(self, call: Any) -> dict:
        if isinstance(call, dict):
            if "function" in call:
                return call
            return {
                "id": call.get("id"),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": self._json_arguments(call.get("args", {})),
                },
            }
        return {
            "id": getattr(call, "id", None),
            "type": "function",
            "function": {
                "name": getattr(call, "name", ""),
                "arguments": self._json_arguments(getattr(call, "args", {})),
            },
        }

    def _json_arguments(self, arguments: Any) -> str:
        import json

        if isinstance(arguments, str):
            return arguments
        return json.dumps(arguments or {}, ensure_ascii=False)

