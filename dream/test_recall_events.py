import datetime as dt

from recall_events import claim_recall_event, finish_recall_event, successful_session_start_ids


def test_event_claim_retry_and_success_suppression(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    assert claim_recall_event(conn, "s1", "session-start:startup", now=t0)
    assert not claim_recall_event(conn, "s1", "session-start:startup", now=t0 + dt.timedelta(seconds=1))
    assert claim_recall_event(conn, "s1", "session-start:startup", now=t0 + dt.timedelta(seconds=11))
    finish_recall_event(conn, "s1", "session-start:startup", succeeded=True, selected_ids=["doc-a"], now=t0 + dt.timedelta(seconds=12))
    assert not claim_recall_event(conn, "s1", "session-start:startup", now=t0 + dt.timedelta(seconds=20))
    assert successful_session_start_ids(conn, "s1") == frozenset({"doc-a"})


def test_failed_event_can_retry_once(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    assert claim_recall_event(conn, "s1", "prompt", now=t0)
    finish_recall_event(conn, "s1", "prompt", succeeded=False, error_code="boom", now=t0)
    assert claim_recall_event(conn, "s1", "prompt", now=t0 + dt.timedelta(seconds=1))
    finish_recall_event(conn, "s1", "prompt", succeeded=False, error_code="boom", now=t0)
    assert not claim_recall_event(conn, "s1", "prompt", now=t0 + dt.timedelta(seconds=2))
