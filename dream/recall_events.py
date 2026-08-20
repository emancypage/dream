"""Exactly-once-ish lifecycle bookkeeping for recall hook events."""

from __future__ import annotations

import datetime as dt
import json

POLICY_VERSION = "recall-v1"
_RETRY_AFTER = dt.timedelta(seconds=10)


def _stamp(value: dt.datetime | None) -> str:
    value = value or dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def claim_recall_event(conn, session_id: str, event: str, *, now: dt.datetime | None = None, policy_version: str = POLICY_VERSION) -> bool:
    """Claim an event, allowing one retry for stale/failed attempts."""
    clock = now or dt.datetime.now(dt.timezone.utc)
    stamp = _stamp(clock)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT status, attempt_count, started_at FROM recall_events WHERE session_id=? AND event=? AND policy_version=?",
            (session_id, event, policy_version),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO recall_events(session_id,event,policy_version,status,attempt_count,started_at) VALUES(?,?,?,?,?,?)",
                (session_id, event, policy_version, "running", 1, stamp),
            )
            conn.commit()
            return True
        status, attempts, started_at = row
        stale = _parse(started_at) is not None and clock - _parse(started_at) >= _RETRY_AFTER
        can_retry = status == "failed" and int(attempts) < 2
        if status == "succeeded" or (status == "running" and not stale) or (status == "failed" and not can_retry):
            conn.commit()
            return False
        if status == "running" and not stale:
            conn.commit()
            return False
        conn.execute(
            "UPDATE recall_events SET status='running', attempt_count=?, started_at=?, finished_at=NULL, error_code=NULL WHERE session_id=? AND event=? AND policy_version=?",
            (int(attempts) + 1, stamp, session_id, event, policy_version),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def finish_recall_event(conn, session_id: str, event: str, *, succeeded: bool, selected_ids=(), error_code: str | None = None, now: dt.datetime | None = None, policy_version: str = POLICY_VERSION) -> None:
    status = "succeeded" if succeeded else "failed"
    conn.execute(
        "UPDATE recall_events SET status=?, finished_at=?, selected_ids_json=?, error_code=? WHERE session_id=? AND event=? AND policy_version=?",
        (status, _stamp(now), json.dumps(list(selected_ids), separators=(",", ":")), error_code, session_id, event, policy_version),
    )
    conn.commit()


def successful_session_start_ids(conn, session_id: str) -> frozenset[str]:
    values: set[str] = set()
    for (raw,) in conn.execute(
        "SELECT selected_ids_json FROM recall_events WHERE session_id=? AND event LIKE 'session-start:%' AND policy_version=? AND status='succeeded'",
        (session_id, POLICY_VERSION),
    ).fetchall():
        try:
            values.update(str(value) for value in json.loads(raw or "[]"))
        except (TypeError, json.JSONDecodeError):
            continue
    return frozenset(values)
