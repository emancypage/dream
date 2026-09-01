from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from sources.base import Message, ParsedSession, TranscriptRef
from sources.common import project_slug


PARSER_VERSION = "opencode-sqlite-v1"
_DB_NAME = "opencode.db"
_VISIBLE_ROLES = {"user", "assistant"}
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def _resolve_db(root: Path) -> Path:
    if root.suffix == ".db" or root.name == _DB_NAME:
        return root
    return root / _DB_NAME


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri()
    sep = "&" if "?" in uri else "?"
    conn = sqlite3.connect(f"{uri}{sep}mode=ro", uri=True, timeout=30.0)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _iso_ms(value: int | None) -> str | None:
    if not value:
        return None
    return dt.datetime.fromtimestamp(value / 1000, tz=dt.timezone.utc).strftime(_ISO)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _revision(session_id: str, *parts: object) -> str:
    payload = "\0".join(str(part or 0) for part in (session_id, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _message_role(raw: str) -> str | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    role = payload.get("role")
    return role if role in _VISIBLE_ROLES else None


def _visible_text(raw: str) -> str | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("type") != "text" or payload.get("synthetic"):
        return None
    text = payload.get("text")
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text or None


class OpenCodeSQLiteSource:
    """Read-only Open Code session adapter backed by opencode.db."""

    def __init__(self, root: Path):
        self.root = root
        self._conn: sqlite3.Connection | None = None
        self._db_path: Path | None = None

    def _open(self) -> tuple[Path, sqlite3.Connection] | None:
        if self._conn is not None and self._db_path is not None:
            return self._db_path, self._conn
        db_path = _resolve_db(self.root)
        if not db_path.is_file():
            return None
        conn = _connect_readonly(db_path)
        names = _table_names(conn)
        missing = {"session", "message", "part"} - names
        if missing:
            conn.close()
            raise RuntimeError(
                "opencode sqlite schema missing tables: " + ", ".join(sorted(missing))
            )
        self._db_path = db_path
        self._conn = conn
        return db_path, conn

    def discover(self) -> Iterable[TranscriptRef]:
        opened = self._open()
        if opened is None:
            return
        db_path, conn = opened
        names = _table_names(conn)
        workspace_join = (
            "LEFT JOIN workspace w ON w.id = s.workspace_id" if "workspace" in names else ""
        )
        rows = conn.execute(
            f"""
            SELECT
                s.id,
                s.time_updated,
                COALESCE(m.cnt, 0),
                COALESCE(m.upd, 0),
                COALESCE(p.cnt, 0),
                COALESCE(p.upd, 0)
            FROM session s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS cnt, MAX(time_updated) AS upd
                FROM message GROUP BY session_id
            ) m ON m.session_id = s.id
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS cnt, MAX(time_updated) AS upd
                FROM part GROUP BY session_id
            ) p ON p.session_id = s.id
            {workspace_join}
            WHERE s.parent_id IS NULL
            ORDER BY s.id
            """
        )
        for session_id, time_updated, msg_count, msg_updated, part_count, part_updated in rows:
            yield TranscriptRef(
                "opencode",
                str(session_id),
                db_path,
                _revision(session_id, time_updated, msg_count, msg_updated, part_count, part_updated),
            )

    def parse(self, ref: TranscriptRef) -> ParsedSession | None:
        opened = self._open()
        if opened is None:
            return None
        db_path, conn = opened
        names = _table_names(conn)
        if "workspace" in names:
            row = conn.execute(
                """
                SELECT s.id, s.parent_id, s.directory, s.time_created, s.time_updated, w.branch
                FROM session s
                LEFT JOIN workspace w ON w.id = s.workspace_id
                WHERE s.id = ?
                """,
                (ref.external_session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT s.id, s.parent_id, s.directory, s.time_created, s.time_updated, NULL
                FROM session s
                WHERE s.id = ?
                """,
                (ref.external_session_id,),
            ).fetchone()
        if row is None or row[1] is not None:
            return None
        _session_id, _parent_id, cwd, started_ms, ended_ms, git_branch = row
        parts_by_message: dict[str, list[str]] = {}
        for message_id, raw in conn.execute(
            """
            SELECT message_id, data
            FROM part
            WHERE session_id = ?
            ORDER BY time_created, id
            """,
            (ref.external_session_id,),
        ):
            text = _visible_text(raw)
            if text:
                parts_by_message.setdefault(message_id, []).append(text)

        messages: list[Message] = []
        for message_id, time_created, raw in conn.execute(
            """
            SELECT id, time_created, data
            FROM message
            WHERE session_id = ?
            ORDER BY time_created, id
            """,
            (ref.external_session_id,),
        ):
            role = _message_role(raw)
            if role is None:
                continue
            text = "\n".join(parts_by_message.get(message_id, [])).strip()
            if not text:
                continue
            messages.append(Message(len(messages) + 1, role, _iso_ms(time_created), text))

        return ParsedSession(
            source="opencode",
            external_session_id=ref.external_session_id,
            revision=ref.revision,
            parser_version=PARSER_VERSION,
            path=db_path,
            project_slug=project_slug(cwd, "opencode"),
            started_at=_iso_ms(started_ms),
            ended_at=_iso_ms(ended_ms),
            cwd=cwd,
            git_branch=git_branch,
            messages=messages,
        )
