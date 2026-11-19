"""Memory compaction contract: which files, the delegate prompt, and the host checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import backend
from config import DelegateSettings, DreamConfig
from consolidate import INDEX_FILE, MemoryFile

MAX_COMPACT_INPUT_CHARS = 200_000
MAX_DROPPED_SUMMARY_CHARS = 2_000
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "delegate-compact.md"
COMPACT_SCHEMA = {
    "type": "object",
    "required": ["body", "dropped_summary", "folded_ids"],
    "additionalProperties": False,
    "properties": {
        "body": {"type": "string"},
        "dropped_summary": {"type": "string", "maxLength": MAX_DROPPED_SUMMARY_CHARS},
        "folded_ids": {"type": "array", "items": {"type": "integer"}},
    },
}
_PLACEHOLDER = re.compile(r"\{\{(file_name|target_chars|file_body|pending_patches)\}\}")


class CompactionError(ValueError):
    """Host-written reason; never contains memory text."""


@dataclass(frozen=True)
class Compaction:
    body: str
    dropped_summary: str
    folded_ids: tuple[int, ...]
    provider: str
    model: str | None


def select_files(files: list[MemoryFile], settings: DelegateSettings) -> list[MemoryFile]:
    over = [entry for entry in files if entry.path.name != INDEX_FILE and settings.compact_over_chars < len(entry.body) <= MAX_COMPACT_INPUT_CHARS]
    return sorted(over, key=lambda entry: len(entry.body), reverse=True)[:settings.compact_max_files_per_run]


def _frontmatter(body: str) -> str | None:
    start = 1 if body.startswith("\ufeff") else 0
    if not body.startswith("---\n", start):
        return None
    opening_end = start + 4
    closing = re.search(r"^---(?:\n|$)", body[opening_end:], re.MULTILINE)
    if closing is None:
        raise CompactionError("invalid frontmatter delimiter")
    return body[:opening_end + closing.end()]


def build_prompt(entry: MemoryFile, patches: list[dict], settings: DelegateSettings, policy_context: str) -> str:
    if len(entry.body) > MAX_COMPACT_INPUT_CHARS:
        raise CompactionError(f"input longer than {MAX_COMPACT_INPUT_CHARS} characters")
    values = {
        "file_name": entry.path.name,
        "target_chars": str(settings.compact_target_chars),
        "file_body": entry.body,
        "pending_patches": json.dumps(patches, ensure_ascii=False, indent=2),
    }
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template) + policy_context


def check_output(entry: MemoryFile, output: dict, patch_ids: set[int],
                 settings: DelegateSettings) -> tuple[str, str, tuple[int, ...]]:
    if len(output["dropped_summary"]) > MAX_DROPPED_SUMMARY_CHARS:
        raise CompactionError(f"dropped_summary longer than {MAX_DROPPED_SUMMARY_CHARS} characters")
    body = output["body"]
    if entry.body.startswith("\ufeff") and not body.startswith("\ufeff"):
        body = "\ufeff" + body
    if not body.removeprefix("\ufeff").strip():
        raise CompactionError("empty body")
    if len(body) > settings.compact_over_chars:
        raise CompactionError("body longer than compact_over_chars")
    if _frontmatter(entry.body) != _frontmatter(body):
        raise CompactionError("frontmatter changed")
    folded = tuple(output["folded_ids"])
    if not set(folded) <= patch_ids:
        raise CompactionError("unknown folded id")
    if len(set(folded)) != len(folded):
        raise CompactionError("duplicate folded id")
    if len(body) < settings.compact_target_chars // 4:
        raise CompactionError("body shorter than a quarter of compact_target_chars")
    return body, output["dropped_summary"], folded


def propose(entry: MemoryFile, patches: list[dict], *, config: DreamConfig, policy_context: str,
            model: str | None = None) -> Compaction:
    settings = config.delegate_settings()
    prompt = build_prompt(entry, patches, settings, policy_context)
    result = backend.generate("delegate", prompt, COMPACT_SCHEMA, model=model, config=config)
    body, dropped, folded = check_output(entry, result.output, {patch["id"] for patch in patches}, settings)
    return Compaction(body, dropped, folded, result.provider, result.model)
