from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sources.base import Message, ParsedSession, TranscriptRef
from sources.common import file_revision, project_slug


PARSER_VERSION = "codex-jsonl-v1"
_INJECTED_PREFIXES = ("# AGENTS.md instructions for", "<environment_context>")


def _is_subagent(meta: dict) -> bool:
    payload = meta.get("payload", {})
    source = payload.get("source")
    return (
        isinstance(source, dict) and "subagent" in source
    ) or payload.get("thread_source") == "subagent"


def _visible_blocks(row: dict, role: str) -> list[str]:
    payload = row.get("payload", {})
    if row.get("type") != "response_item" or payload.get("type") != "message":
        return []
    if payload.get("role") != role:
        return []
    if role == "assistant" and payload.get("phase") != "final_answer":
        return []
    block_type = "input_text" if role == "user" else "output_text"
    parts = []
    for block in payload.get("content", []):
        if not isinstance(block, dict) or block.get("type") != block_type:
            continue
        text = str(block.get("text", "")).strip()
        if not text or (role == "user" and text.lstrip().startswith(_INJECTED_PREFIXES)):
            continue
        parts.append(text)
    return parts


class CodexJSONLSource:
    def __init__(self, root: Path):
        self.root = root

    def discover(self) -> Iterable[TranscriptRef]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.jsonl")):
            meta = self._first_meta(path)
            if not meta or _is_subagent(meta):
                continue
            payload = meta.get("payload", {})
            external_id = payload.get("session_id") or payload.get("id") or path.stem
            yield TranscriptRef("codex", str(external_id), path, file_revision(path))

    @staticmethod
    def _first_meta(path: Path) -> dict | None:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") == "session_meta":
                    return row
        return None

    def parse(self, ref: TranscriptRef) -> ParsedSession | None:
        meta = self._first_meta(ref.path)
        if not meta or _is_subagent(meta):
            return None
        payload = meta.get("payload", {})
        cwd = payload.get("cwd")
        messages: list[Message] = []
        seen_ids: set[str] = set()
        started = ended = meta.get("timestamp") or payload.get("timestamp")
        with ref.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp = row.get("timestamp")
                if timestamp:
                    started = started or timestamp
                    ended = timestamp
                item_id = row.get("payload", {}).get("id")
                if item_id and item_id in seen_ids:
                    continue
                user_parts = _visible_blocks(row, "user")
                assistant_parts = _visible_blocks(row, "assistant")
                if not user_parts and not assistant_parts:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                role = "user" if user_parts else "assistant"
                text = "\n".join(user_parts or assistant_parts).strip()
                if text:
                    messages.append(Message(len(messages) + 1, role, timestamp, text))
        return ParsedSession(
            source="codex",
            external_session_id=ref.external_session_id,
            revision=ref.revision,
            parser_version=PARSER_VERSION,
            path=ref.path,
            project_slug=project_slug(cwd, "codex"),
            started_at=started,
            ended_at=ended,
            cwd=cwd,
            git_branch=None,
            messages=messages,
        )
