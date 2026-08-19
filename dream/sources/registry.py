from __future__ import annotations

from pathlib import Path

from config import ConfigError
from sources.base import TranscriptSource


def create_source(source_type: str, root: Path) -> TranscriptSource:
    if source_type == "codex-jsonl":
        from sources.codex_jsonl import CodexJSONLSource

        return CodexJSONLSource(root)
    if source_type == "claude-jsonl":
        from sources.claude_jsonl import ClaudeJSONLSource

        return ClaudeJSONLSource(root)
    raise ConfigError(f"unsupported source type: {source_type}")
