"""Tests for consolidate's incremental batch selection (collect_distilled_batch).

The selection anchors on a session's *last activity*, not its start. This matters
for resumed sessions: `dream ingest` re-imports a resumed JSONL (mtime grew) and the
FK `ON DELETE CASCADE` on `distilled` wipes the stale distillation, so the session
gets re-distilled over the full (old+new) transcript. But its `started_at` is frozen
at the first message's timestamp — for a long-ago session that sits *below* the
consolidate watermark. Anchoring the batch filter on `started_at` would strand that
fresh distillation forever. Anchoring on `ended_at` (which the resume bumps to "now")
lets the continued thread flow into memory, while genuinely untouched sessions stay out.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consolidate import collect_distilled_batch  # noqa: E402

SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

WATERMARK = "2026-06-04T15:14:41.366Z"


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def _session(conn, sid, started, ended, *, distilled=True, chars=2000):
    conn.execute(
        "INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime, "
        "started_at, ended_at, total_chars) VALUES (?, '-home-user', '/p.jsonl', 1.0, ?, ?, ?)",
        (sid, started, ended, chars),
    )
    if distilled:
        conn.execute(
            "INSERT INTO distilled(session_id, notes_json, summary) VALUES (?, ?, 's')",
            (sid, json.dumps({"facts": [f"fact from {sid}"]})),
        )
    conn.commit()


def _ids(batch):
    return [row[0] for row in batch]


def test_resumed_old_session_is_recollected():
    # started long before the watermark, but resumed today -> ended_at is now.
    conn = _db()
    _session(conn, "resumed-old", started="2026-06-01T10:00:00", ended="2026-06-05T18:00:00")
    batch = collect_distilled_batch(conn, WATERMARK)
    assert _ids(batch) == ["resumed-old"]


def test_already_consolidated_untouched_session_stays_out():
    # both timestamps below the watermark, never resumed -> must NOT re-consolidate.
    conn = _db()
    _session(conn, "old-done", started="2026-06-01T10:00:00", ended="2026-06-01T10:30:00")
    assert collect_distilled_batch(conn, WATERMARK) == []


def test_batch_ts_anchors_on_ended_at_so_watermark_advances():
    # the ts the watermark advances to must be the resume's end, not the old start.
    conn = _db()
    _session(conn, "resumed-old", started="2026-06-01T10:00:00", ended="2026-06-05T18:00:00")
    batch = collect_distilled_batch(conn, WATERMARK)
    assert batch[-1][1] == "2026-06-05T18:00:00"


def test_ordering_is_by_last_activity():
    conn = _db()
    _session(conn, "a", started="2026-05-01T00:00:00", ended="2026-06-05T09:00:00")
    _session(conn, "b", started="2026-06-04T20:00:00", ended="2026-06-04T20:10:00")
    # 'a' started earlier but ended later -> sorts AFTER 'b'.
    assert _ids(collect_distilled_batch(conn, WATERMARK)) == ["b", "a"]


def test_chat_session_unaffected():
    # chat_ingest sets started_at == ended_at == last ts; behaviour is unchanged.
    conn = _db()
    _session(conn, "chat-1-9", started="2026-06-05T12:00:00", ended="2026-06-05T12:00:00")
    assert _ids(collect_distilled_batch(conn, WATERMARK)) == ["chat-1-9"]
