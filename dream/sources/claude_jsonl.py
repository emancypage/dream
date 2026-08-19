from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sources.base import Message, ParsedSession, TranscriptRef
from sources.common import file_revision, project_slug


PARSER_VERSION = "claude-jsonl-v2"


def _user_text(row: dict) -> str | None:
    content = row.get("message", {}).get("content")
    if isinstance(content, str):
        if content.startswith("<command-name>") or content.startswith("<local-command-stdout>"):
            return None
        return content.strip() or None
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(filter(None, parts)).strip() or None
    return None


def _assistant_text(row: dict) -> str | None:
    content = row.get("message", {}).get("content")
    if not isinstance(content, list):
        return None
    parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    return "\n".join(filter(None, parts)).strip() or None


class ClaudeJSONLSource:
    def __init__(self, root: Path):
        self.root = root

    def discover(self) -> Iterable[TranscriptRef]:
        if not self.root.exists():
            return
        for project_dir in sorted(self.root.iterdir()):
            if not project_dir.is_dir():
                continue
            for path in sorted(project_dir.glob("*.jsonl")):
                yield TranscriptRef("claude", path.stem, path, file_revision(path))

    def parse(self, ref: TranscriptRef) -> ParsedSession:
        messages: list[Message] = []
        cwd = branch = started = ended = None
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
                cwd = cwd or row.get("cwd")
                branch = branch or row.get("gitBranch")
                if row.get("isSidechain"):
                    continue
                text = None
                role = row.get("type")
                if role == "user":
                    text = _user_text(row)
                elif role == "assistant":
                    text = _assistant_text(row)
                if text:
                    messages.append(Message(len(messages) + 1, role, timestamp, text))
        return ParsedSession(
            source="claude",
            external_session_id=ref.external_session_id,
            revision=ref.revision,
            parser_version=PARSER_VERSION,
            path=ref.path,
            project_slug=project_slug(cwd, ref.path.parent.name),
            started_at=started,
            ended_at=ended,
            cwd=cwd,
            git_branch=branch,
            messages=messages,
        )
