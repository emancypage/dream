"""Provider-independent transcript ingestion into SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from config import DreamConfig, load_config
from sources.base import Message, ParsedSession
from sources.claude_jsonl import ClaudeJSONLSource
from sources.registry import create_source


def home_project_slug() -> str:
    return str(Path.home()).replace("/", "-")


def _internal_id(conn: sqlite3.Connection, parsed: ParsedSession) -> str:
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE source=? AND external_session_id=?",
        (parsed.source, parsed.external_session_id),
    ).fetchone()
    if row:
        return row[0]
    if parsed.source == "claude":
        collision = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id=?", (parsed.external_session_id,)
        ).fetchone()
        if not collision:
            return parsed.external_session_id
    return f"{parsed.source}:{parsed.external_session_id}"


def ingest_parsed_session(
    parsed: ParsedSession,
    conn: sqlite3.Connection,
    force: bool = False,
) -> tuple[bool, int]:
    session_id = _internal_id(conn, parsed)
    existing = conn.execute(
        "SELECT source_revision, parser_version FROM sessions WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if not force and existing == (parsed.revision, parsed.parser_version):
        return False, 0

    with conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute(
            """
            INSERT INTO sessions(
                session_id, source, external_session_id, source_revision, parser_version,
                project_slug, jsonl_path, jsonl_mtime, started_at, ended_at, cwd,
                git_branch, user_msg_count, asst_msg_count, total_chars, ingested_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                source=excluded.source,
                external_session_id=excluded.external_session_id,
                source_revision=excluded.source_revision,
                parser_version=excluded.parser_version,
                project_slug=excluded.project_slug,
                jsonl_path=excluded.jsonl_path,
                jsonl_mtime=excluded.jsonl_mtime,
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                cwd=excluded.cwd,
                git_branch=excluded.git_branch,
                user_msg_count=excluded.user_msg_count,
                asst_msg_count=excluded.asst_msg_count,
                total_chars=excluded.total_chars,
                ingested_at=CURRENT_TIMESTAMP
            """,
            (
                session_id, parsed.source, parsed.external_session_id, parsed.revision,
                parsed.parser_version, parsed.project_slug, str(parsed.path),
                parsed.path.stat().st_mtime, parsed.started_at, parsed.ended_at, parsed.cwd,
                parsed.git_branch, parsed.user_msg_count, parsed.asst_msg_count,
                parsed.total_chars,
            ),
        )
        conn.executemany(
            "INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES (?, ?, ?, ?, ?)",
            [(session_id, message.seq, message.role, message.timestamp, message.text) for message in parsed.messages],
        )
    return True, len(parsed.messages)


def ingest_source(source, conn: sqlite3.Connection, force: bool = False) -> tuple[int, int, int]:
    seen = ingested = messages = 0
    for ref in source.discover():
        seen += 1
        parsed = source.parse(ref)
        if parsed is None:
            continue
        changed, count = ingest_parsed_session(parsed, conn, force=force)
        if changed:
            ingested += 1
            messages += count
    return seen, ingested, messages


def ingest_configured_sources(
    conn: sqlite3.Connection,
    config: DreamConfig | None = None,
    selected: set[str] | None = None,
    force: bool = False,
) -> list[tuple[str, int, int, int]]:
    cfg = config or load_config()
    results = []
    for source_cfg in cfg.sources:
        if not source_cfg.get("enabled", True) and not (selected and source_cfg["name"] in selected):
            continue
        if selected and source_cfg["name"] not in selected:
            continue
        source = create_source(source_cfg["type"], Path(source_cfg["root"]).expanduser())
        results.append((source_cfg["name"], *ingest_source(source, conn, force=force)))
    return results


# Compatibility functions used by the old CLI and external callers.
def discover_jsonl(projects_root: Path) -> Iterator[Path]:
    source = ClaudeJSONLSource(projects_root)
    for ref in source.discover():
        yield ref.path


def parse_jsonl(path: Path) -> Iterator[Message]:
    source = ClaudeJSONLSource(path.parent.parent)
    from sources.common import file_revision
    from sources.base import TranscriptRef

    ref = TranscriptRef("claude", path.stem, path, file_revision(path))
    yield from source.parse(ref).messages


def ingest_file(path: Path, conn: sqlite3.Connection, force: bool = False) -> tuple[bool, int]:
    from sources.common import file_revision
    from sources.base import TranscriptRef

    source = ClaudeJSONLSource(path.parent.parent)
    ref = TranscriptRef("claude", path.stem, path, file_revision(path))
    return ingest_parsed_session(source.parse(ref), conn, force=force)
