"""
Full-text search across ingested transcripts using SQLite FTS5.

Designed for the `/dream-search <query>` slash command: returns a compact list
of matches with session, role, timestamp, and a snippet with highlights.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class SearchHit:
    session_id: str
    role: str
    timestamp: str | None
    snippet: str
    project_slug: str
    cwd: str | None
    score: float


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    role: str | None = None,
    project_slug: str | None = None,
) -> list[SearchHit]:
    """
    FTS5 MATCH search. Returns ranked snippets (best first).

    `query` is passed straight to FTS5 — supports quoted phrases, NEAR(),
    column filters, etc. See https://sqlite.org/fts5.html#full_text_query_syntax
    """
    sql = """
        SELECT
            m.session_id,
            m.role,
            m.timestamp,
            snippet(messages_fts, 0, '«', '»', '…', 16) AS snip,
            s.project_slug,
            s.cwd,
            bm25(messages_fts) AS score
        FROM messages_fts
        JOIN messages m ON m.rowid = messages_fts.rowid
        JOIN sessions s ON s.session_id = m.session_id
        WHERE messages_fts MATCH ?
    """
    params: list = [query]
    if role:
        sql += " AND m.role = ?"
        params.append(role)
    if project_slug:
        sql += " AND s.project_slug = ?"
        params.append(project_slug)
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)

    out = []
    for row in conn.execute(sql, params).fetchall():
        out.append(SearchHit(
            session_id=row[0],
            role=row[1],
            timestamp=row[2],
            snippet=row[3],
            project_slug=row[4],
            cwd=row[5],
            score=row[6],
        ))
    return out


def session_summary(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute("""
        SELECT s.session_id, s.project_slug, s.cwd, s.git_branch, s.started_at, s.ended_at,
               s.user_msg_count, s.asst_msg_count, s.total_chars, d.summary
        FROM sessions s
        LEFT JOIN distilled d ON d.session_id = s.session_id
        WHERE s.session_id = ?
    """, (session_id,)).fetchone()
    if not row:
        return None
    return {
        "session_id": row[0], "project_slug": row[1], "cwd": row[2], "git_branch": row[3],
        "started_at": row[4], "ended_at": row[5],
        "user_msg_count": row[6], "asst_msg_count": row[7], "total_chars": row[8],
        "distilled_summary": row[9],
    }
