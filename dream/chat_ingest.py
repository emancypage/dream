"""Optional bridge: fold an external chat log into dream's session pipeline.

A companion app can write conversation turns into a `chat_messages` table inside the
shared ~/.claude/dream.db. dream otherwise never sees them — it only ingests Claude
Code JSONL transcripts. This module materializes accumulated chat turns as a synthetic
dream *session* (rows in `sessions` + `messages`), so the existing distill ->
consolidate pipeline folds a conversation into long-term memory exactly the way it
folds a coding session. No changes to distill/consolidate needed.

The bridge is inert when the table is absent: `ingest_chat` returns None, so a plain
`dream ingest` on a stock install is unaffected.

Design (three correctness invariants):
  * Watermark `meta['last_chat_ingest_id']` tracks the last chat_messages.id already
    folded into a session. We advance it ONLY when we actually create a session, so a
    short burst of small talk just accumulates until it's worth distilling — and never
    gets advanced past below distill's char floor (which would lose it silently).
  * The synthetic session's `started_at` is the LAST message's timestamp, not the
    first. consolidate selects distilled rows where COALESCE(started_at, distilled_at)
    is greater than its own watermark; anchoring to the newest message guarantees the
    session sorts AFTER any consolidate that already ran (no distilled-but-never-
    consolidated gap).
  * `min_chars` defaults to distill's own floor (500), so every session we create is
    guaranteed to clear `sessions_needing_distill` — we never strand a bundle.

Both roles are kept for context. Safe because consolidate is suggest-only: every derived
fact passes `dream review` before touching live memory. The session_id prefix makes the
chat provenance visible in review.
"""
from __future__ import annotations

import sqlite3

from ingest import home_project_slug

# Same bucket Claude Code uses for sessions started in $HOME, so chat-derived memory
# lands alongside the user's other personal context during consolidation.
CHAT_PROJECT_SLUG = home_project_slug()

# Match distill's default min_chars floor: a session we create must be distillable.
DEFAULT_MIN_CHARS = 500


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _get_watermark(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='last_chat_ingest_id'").fetchone()
    try:
        return int(row[0]) if row and row[0] is not None else 0
    except (TypeError, ValueError):
        return 0


def set_watermark(conn: sqlite3.Connection, last_id: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_chat_ingest_id', ?)",
        (str(int(last_id)),),
    )
    conn.commit()


def ingest_chat(
    conn: sqlite3.Connection,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    project_slug: str = CHAT_PROJECT_SLUG,
) -> str | None:
    """Fold new chat_messages into one synthetic session. Return its id, or None.

    No-op (watermark untouched) when there are no new rows, the chat table is absent
    (dev db without the companion app), or the combined text is below `min_chars` — small talk
    accumulates until it's substantial enough to distill.
    """
    if not _table_exists(conn, "chat_messages"):
        return None

    watermark = _get_watermark(conn)
    rows = conn.execute(
        "SELECT id, role, text, ts FROM chat_messages WHERE id > ? ORDER BY id",
        (watermark,),
    ).fetchall()
    if not rows:
        return None

    total_chars = sum(len(r[2] or "") for r in rows)
    if total_chars < min_chars:
        return None  # leave watermark — let the conversation accumulate

    first_id, last_id = rows[0][0], rows[-1][0]
    last_ts = rows[-1][3]
    session_id = f"chat-{first_id}-{last_id}"

    user_count = sum(1 for r in rows if r[1] == "user")
    asst_count = len(rows) - user_count

    conn.execute(
        """
        INSERT OR REPLACE INTO sessions(
            session_id, project_slug, jsonl_path, jsonl_mtime,
            started_at, ended_at, cwd, git_branch,
            user_msg_count, asst_msg_count, total_chars
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id, project_slug, "(chat-bridge)", 0.0,
            last_ts, last_ts, "(chat)", None,
            user_count, asst_count, total_chars,
        ),
    )
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.executemany(
        "INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES (?, ?, ?, ?, ?)",
        [
            (session_id, seq, ("user" if r[1] == "user" else "assistant"), r[3], r[2])
            for seq, r in enumerate(rows)
        ],
    )
    set_watermark(conn, last_id)  # commits
    return session_id
