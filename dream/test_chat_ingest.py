"""Tests for the external-chat → dream-session bridge (chat_ingest.ingest_chat).

These encode the three correctness traps the design hinges on:
  * started_at = the bundle's LAST message ts (so it sorts AFTER any prior consolidate)
  * sub-threshold small talk accumulates (watermark untouched) — never silently dropped
  * a created session is guaranteed >= distill's min_chars floor, so it actually distills
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import chat_ingest  # noqa: E402
from distill import sessions_needing_distill  # noqa: E402

SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _db(with_chat_table: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    if with_chat_table:
        # the companion app owns this table in the real db; recreate its shape for the test.
        conn.execute(
            "CREATE TABLE chat_messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, role TEXT NOT NULL, text TEXT NOT NULL)"
        )
    return conn


def _add(conn: sqlite3.Connection, role: str, text: str, ts: str) -> None:
    conn.execute("INSERT INTO chat_messages(role, text, ts) VALUES (?, ?, ?)", (role, text, ts))
    conn.commit()


def _watermark(conn: sqlite3.Connection):
    row = conn.execute("SELECT value FROM meta WHERE key='last_chat_ingest_id'").fetchone()
    return row[0] if row else None


def test_noop_when_no_messages():
    conn = _db()
    assert chat_ingest.ingest_chat(conn) is None


def test_small_talk_accumulates_watermark_untouched():
    conn = _db()
    _add(conn, "user", "siema", "2026-06-05T10:00:00")
    _add(conn, "bot", "no elo", "2026-06-05T10:00:05")
    assert chat_ingest.ingest_chat(conn, min_chars=400) is None
    assert _watermark(conn) is None              # nothing advanced — accumulate next time
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_creates_session_when_substantial():
    conn = _db()
    _add(conn, "user", "x" * 250, "2026-06-05T10:00:00")
    _add(conn, "bot", "y" * 250, "2026-06-05T10:01:00")
    sid = chat_ingest.ingest_chat(conn, min_chars=400)
    assert sid == "chat-1-2"
    project, started, ended, total, ucnt, acnt = conn.execute(
        "SELECT project_slug, started_at, ended_at, total_chars, user_msg_count, asst_msg_count "
        "FROM sessions WHERE session_id=?", (sid,)
    ).fetchone()
    assert project == chat_ingest.CHAT_PROJECT_SLUG
    assert started == "2026-06-05T10:01:00"      # TRAP: ordering ts = LAST msg, not first
    assert ended == "2026-06-05T10:01:00"
    assert total == 500
    assert (ucnt, acnt) == (1, 1)
    roles = [r[0] for r in conn.execute(
        "SELECT role FROM messages WHERE session_id=? ORDER BY seq", (sid,))]
    assert roles == ["user", "assistant"]        # any non-user role -> assistant (dream's role vocab)


def test_created_session_is_picked_up_by_distill():
    conn = _db()
    _add(conn, "user", "a" * 300, "2026-06-05T10:00:00")
    _add(conn, "bot", "b" * 300, "2026-06-05T10:01:00")
    sid = chat_ingest.ingest_chat(conn, min_chars=400)
    pending = [s for s, _ in sessions_needing_distill(conn, min_chars=500)]
    assert sid in pending                        # TRAP: clears distill's floor -> not dropped


def test_watermark_advances_and_second_run_is_noop():
    conn = _db()
    _add(conn, "user", "a" * 500, "2026-06-05T10:00:00")
    sid = chat_ingest.ingest_chat(conn, min_chars=400)
    assert sid is not None
    assert _watermark(conn) == "1"
    assert chat_ingest.ingest_chat(conn, min_chars=400) is None   # no new rows


def test_only_new_messages_after_watermark_are_bundled():
    conn = _db()
    _add(conn, "user", "a" * 500, "2026-06-05T10:00:00")
    chat_ingest.ingest_chat(conn, min_chars=400)                  # consumes id 1
    _add(conn, "user", "b" * 500, "2026-06-06T10:00:00")          # id 2, new day
    sid2 = chat_ingest.ingest_chat(conn, min_chars=400)
    assert sid2 == "chat-2-2"
    seqs = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (sid2,)).fetchone()[0]
    assert seqs == 1                                             # only the new message


def test_missing_chat_messages_table_is_graceful():
    conn = _db(with_chat_table=False)
    assert chat_ingest.ingest_chat(conn) is None                # dev db without the companion app -> no crash
