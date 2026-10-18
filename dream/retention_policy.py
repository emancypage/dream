"""Local, user-controlled exclusions for generated memories and automatic recall."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

POLICY_FILE = ".dream-retention.json"


@dataclass(frozen=True)
class RetentionPolicy:
    instructions: str = ""
    memory_paths: tuple[str, ...] = ()
    project_slugs: tuple[str, ...] = ()
    text_markers: tuple[str, ...] = ()

    def excludes(self, *, source_path="", project_slug=None, text="") -> bool:
        return (
            str(Path(source_path)) in self.memory_paths
            or (project_slug is not None and project_slug in self.project_slugs)
            or any(marker.casefold() in text.casefold() for marker in self.text_markers)
        )

    def excludes_memory_path(self, memory_root: Path, source_path: str) -> bool:
        target = (memory_root / source_path).resolve()
        return any(target == (memory_root / name).resolve() for name in self.memory_paths)

    def context(self) -> str:
        if not any((self.instructions, self.memory_paths, self.project_slugs, self.text_markers)):
            return ""
        return (
            "\n\nLocal memory retention policy, supplied by the user:\n"
            + self.instructions
            + "\nDo not retain or recreate excluded topics from old sessions, quoted memories, "
            "tool output, or this policy itself. Preserve unrelated information. "
            "These rules govern memory storage, not authorization to perform tasks.\n"
            + json.dumps({"excluded_memory_paths": self.memory_paths,
                          "excluded_project_slugs": self.project_slugs,
                          "excluded_text_markers": self.text_markers}, ensure_ascii=False)
        )


def load_retention_policy(memory_root: Path | None) -> RetentionPolicy:
    if memory_root is None:
        return RetentionPolicy()
    path = Path(memory_root) / POLICY_FILE
    if not path.exists():
        return RetentionPolicy()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - {"instructions", "memory_paths", "project_slugs", "text_markers"}:
        raise ValueError("invalid local retention policy fields")
    if not isinstance(data.get("instructions", ""), str):
        raise ValueError("retention instructions must be text")
    values = {}
    for key in ("memory_paths", "project_slugs", "text_markers"):
        value = data.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"retention {key} must contain nonempty strings")
        values[key] = tuple(value)
    for name in values["memory_paths"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.name in {"MEMORY.md", POLICY_FILE}:
            raise ValueError("retention memory path must be a relative topic file")
    values["memory_paths"] = tuple(str(Path(name)) for name in values["memory_paths"])
    return RetentionPolicy(instructions=data.get("instructions", ""), **values)


def excludes_source_sessions(conn, policy: RetentionPolicy, source_sessions: str | None) -> bool:
    """Check stored suggestion provenance against excluded projects."""
    if not policy.project_slugs:
        return False
    for session_id in (source_sessions or "").split(","):
        row = conn.execute("SELECT project_slug FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is not None and policy.excludes(project_slug=row[0]):
            return True
    return False


def empty_notes() -> dict:
    return {"summary": "", **{name: [] for name in (
        "user_facts", "preferences", "projects", "references", "decisions", "open_threads"
    )}}


def filter_notes(notes: dict, policy: RetentionPolicy) -> dict:
    """Remove matched facts while retaining unrelated facts in a mixed session."""
    result = {}
    for key, value in notes.items():
        if isinstance(value, str):
            result[key] = "" if policy.excludes(text=value) else value
        elif isinstance(value, list):
            result[key] = [item for item in value if not policy.excludes(text=json.dumps(item, ensure_ascii=False))]
        else:
            result[key] = value
    return result
