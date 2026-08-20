"""Transactional synchronization of recall documents.

Collects approved memory files, distilled summaries, and (optionally) raw
transcripts into plain row lists, then in one write transaction upserts
changed rows, removes stale rows for the source kinds that collected
successfully, and repairs the external-content FTS index when it is missing,
corrupt, or changed. Any collection, write, or FTS error rolls back and
retains the previous complete database state. This module never prints to
stdout.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

# One fixed namespace for every stable document ID, regardless of source kind.
_DOCUMENT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "dream-recall-documents")

_MEMORY_EXCLUDED_DIRS = {".suggestions", "memory-backups"}
_DISTILLED_TEXT_MAX_CHARS = 200_000


@dataclass
class SyncReport:
    """Counters describing one synchronization pass."""

    scanned: int = 0
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    raw_included: int = 0


def stable_document_id(kind: str, locator: str) -> str:
    """Return the stable UUIDv5 identity for one (kind, locator) pair."""
    return str(uuid.uuid5(_DOCUMENT_NAMESPACE, f"{kind}:{locator}"))


def _format_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _file_timestamp(path: Path) -> str:
    try:
        value = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    except OSError:
        value = dt.datetime.now(dt.timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _collect_memory_files(memory_root: Path) -> list[dict]:
    """Collect approved memory files.

    Hidden/transient files, .suggestions/, and memory-backups/ are excluded;
    non-UTF-8 files are skipped without aborting the run. Walks the tree with
    plain OS primitives so the root never needs to be a real ``Path`` (tests
    may pass narrow fakes to simulate mid-walk failures).
    """
    rows: list[dict] = []
    if not memory_root.is_dir():
        return rows

    def _is_excluded_dir(name: str) -> bool:
        return name in _MEMORY_EXCLUDED_DIRS or name.startswith(".")

    def _walk(directory, prefix: str) -> None:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
        for entry in entries:
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            if entry.name.endswith(".tmp") or entry.name.endswith("~"):
                continue
            try:
                text = Path(entry.path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel_path = f"{prefix}{entry.name}"
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            doc_id = stable_document_id("approved_memory", f"memory:{rel_path}")
            rows.append({
                "id": doc_id,
                "content_sha256": digest,
                "source_kind": "approved_memory",
                "trust_level": "user_approved",
                "project_slug": None,
                "source_path": rel_path,
                "source_updated_at": None,  # filled with the synchronization clock
                "source_version": digest,
                "text": text,
            })
        for entry in entries:
            if entry.is_dir() and not _is_excluded_dir(entry.name):
                _walk(entry, f"{prefix}{entry.name}/")

    _walk(memory_root, "")
    return rows


def _collect_distilled(conn: sqlite3.Connection) -> list[dict]:
    rows: list[dict] = []
    for (key, notes_json, summary, project_slug, distilled_at) in conn.execute(
        """
        SELECT COALESCE(d.distillation_key, 'legacy:' || d.session_id) AS key,
               d.notes_json,
               d.summary,
               s.project_slug,
               d.distilled_at
        FROM distilled d
        JOIN sessions s ON s.session_id = d.session_id
        """
    ).fetchall():
        if not key:
            continue
        try:
            notes = json.loads(notes_json) if notes_json else None
        except (TypeError, json.JSONDecodeError):
            notes = None
        notes_present = isinstance(notes, (dict, list)) and bool(notes)
        if notes_present:
            text = json.dumps(notes, ensure_ascii=False, sort_keys=True)
        else:
            text = summary or ""
        if not text:
            continue
        text = text[:_DISTILLED_TEXT_MAX_CHARS]
        updated_at = _format_timestamp(distilled_at) or "1970-01-01T00:00:00Z"
        locator = f"distilled:{key}"
        rows.append({
            "id": stable_document_id("distilled_summary", locator),
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_kind": "distilled_summary",
            "trust_level": "model_distilled",
            "project_slug": project_slug,
            "source_path": locator,
            "source_updated_at": updated_at,
            "source_version": key,
            "text": text,
        })
    return rows


def _render_transcript(messages: list[tuple[str, str, str | None]]) -> str:
    lines: list[str] = []
    for role, text, timestamp in messages:
        label = "USER" if role == "user" else "ASSISTANT"
        header = f"--- {label}"
        if timestamp:
            header += f" [{timestamp}]"
        header += " ---"
        lines.append(header)
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def _collect_transcripts(conn: sqlite3.Connection) -> list[dict]:
    rows: list[dict] = []
    for (session_id, source, external_id, project_slug,
         source_revision, started_at) in conn.execute(
        """
        SELECT s.session_id, s.source, s.external_session_id, s.project_slug,
               s.source_revision, s.started_at
        FROM sessions s
        ORDER BY s.session_id
        """
    ).fetchall():
        external_id = external_id or session_id
        locator = f"transcript:{source}:{external_id}"
        messages = conn.execute(
            "SELECT role, text, timestamp FROM messages WHERE session_id=? "
            "ORDER BY seq",
            (session_id,),
        ).fetchall()
        if not messages:
            continue
        text = _render_transcript(list(messages))
        if not text:
            continue
        updated_at = _format_timestamp(started_at) or "1970-01-01T00:00:00Z"
        rows.append({
            "id": stable_document_id("raw_transcript", locator),
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_kind": "raw_transcript",
            "trust_level": "untrusted_transcript",
            "project_slug": project_slug,
            "source_path": locator,
            "source_updated_at": updated_at,
            "source_version": source_revision or "",
            "text": text,
        })
    return rows


def _upsert_document(conn: sqlite3.Connection, row: dict, indexed_at: str) -> None:
    conn.execute(
        """
        INSERT INTO recall_documents(
            id, content_sha256, source_kind, trust_level, project_slug,
            source_path, source_updated_at, indexed_at, source_version, text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            content_sha256=excluded.content_sha256,
            source_kind=excluded.source_kind,
            trust_level=excluded.trust_level,
            project_slug=excluded.project_slug,
            source_path=excluded.source_path,
            source_updated_at=excluded.source_updated_at,
            indexed_at=excluded.indexed_at,
            source_version=excluded.source_version,
            text=excluded.text
        """,
        (
            row["id"], row["content_sha256"], row["source_kind"],
            row["trust_level"], row["project_slug"], row["source_path"],
            row["source_updated_at"], indexed_at, row["source_version"],
            row["text"],
        ),
    )


def _fts_index_state(conn: sqlite3.Connection) -> tuple[bool, bool] | None:
    """Return ``(present, healthy)``; None when the index state is unknown."""
    row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name='recall_documents_fts'"
    ).fetchone()
    if row is None:
        return False, False
    table_rows = conn.execute("SELECT COUNT(*) FROM recall_documents").fetchone()[0]
    try:
        index_rows = conn.execute(
            "SELECT COUNT(*) FROM recall_documents_fts"
        ).fetchone()[0]
    except sqlite3.Error:
        return None
    if index_rows != table_rows:
        return True, False
    try:
        conn.execute("INSERT INTO recall_documents_fts(recall_documents_fts) VALUES ('integrity-check')")
    except sqlite3.Error:
        return True, False
    return True, True


def synchronize_recall_documents(
    conn: sqlite3.Connection,
    memory_root: Path,
    *,
    include_raw_transcripts: bool,
    now: dt.datetime | None = None,
) -> SyncReport:
    """Synchronize recall_documents from every configured source.

    All source rows are collected before the write transaction opens. In one
    transaction the upsert of changed rows, the deletion of stale rows (only
    for source kinds whose collection completed successfully), and the FTS
    index repair happen together; any error rolls back the entire pass and
    retains the previous complete database state.
    """
    indexed_at = (
        now.astimezone(dt.timezone.utc)
        if now is not None
        else dt.datetime.now(dt.timezone.utc)
    ).isoformat()

    completed: set[str] = set()
    memory_rows: list[dict] = []
    distilled_rows: list[dict] = []
    transcript_rows: list[dict] = []

    memory_rows = _collect_memory_files(memory_root)
    for row in memory_rows:
        row["source_updated_at"] = indexed_at
    completed.add("approved_memory")

    distilled_rows = _collect_distilled(conn)
    completed.add("distilled_summary")

    if include_raw_transcripts:
        transcript_rows = _collect_transcripts(conn)
        completed.add("raw_transcript")

    existing: dict[str, dict] = {
        row[0]: {
            "content_sha256": row[1],
            "source_kind": row[2],
            "trust_level": row[3],
            "project_slug": row[4],
            "source_path": row[5],
            "source_updated_at": row[6],
            "indexed_at": row[7],
            "source_version": row[8],
            "text": row[9],
        }
        for row in conn.execute(
            "SELECT id, content_sha256, source_kind, trust_level, project_slug, "
            "source_path, source_updated_at, indexed_at, source_version, text "
            "FROM recall_documents"
        ).fetchall()
    }

    changed: list[dict] = []
    skipped = 0
    for row in (*memory_rows, *distilled_rows, *transcript_rows):
        prior = existing.get(row["id"])
        if prior is None:
            changed.append(row)
            continue
        if (
            prior["content_sha256"] == row["content_sha256"]
            and prior["source_kind"] == row["source_kind"]
            and prior["trust_level"] == row["trust_level"]
            and prior["project_slug"] == row["project_slug"]
            and prior["source_path"] == row["source_path"]
            and prior["source_version"] == row["source_version"]
        ):
            skipped += 1
        else:
            changed.append(row)

    fresh_by_kind: dict[str, set[str]] = {}
    if "approved_memory" in completed:
        fresh_by_kind["approved_memory"] = {row["id"] for row in memory_rows}
    if "distilled_summary" in completed:
        fresh_by_kind["distilled_summary"] = {row["id"] for row in distilled_rows}
    if "raw_transcript" in completed:
        fresh_by_kind["raw_transcript"] = {row["id"] for row in transcript_rows}

    stale_ids = [
        doc_id
        for doc_id, prior in existing.items()
        if (
            prior["source_kind"] in fresh_by_kind
            and doc_id not in fresh_by_kind[prior["source_kind"]]
        )
    ]

    with conn:
        for row in changed:
            _upsert_document(conn, row, indexed_at)
        for doc_id in stale_ids:
            conn.execute("DELETE FROM recall_documents WHERE id=?", (doc_id,))
        state = _fts_index_state(conn)
        if state is None or not state[0] or not state[1] or changed or stale_ids:
            # Imported lazily: dream.py imports this module at load time.
            from dream import rebuild_recall_fts

            rebuild_recall_fts(conn)

    inserted = sum(1 for row in changed if row["id"] not in existing)
    updated = len(changed) - inserted

    return SyncReport(
        scanned=len(memory_rows) + len(distilled_rows) + len(transcript_rows),
        inserted=inserted,
        updated=updated,
        deleted=len(stale_ids),
        skipped=skipped,
        raw_included=len(transcript_rows),
    )
