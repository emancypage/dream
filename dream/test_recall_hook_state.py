import json
from datetime import datetime, timedelta, timezone

from recall_hook_state import claim_hook_event, finish_hook_event, successful_session_start_ids


def _make_running_marker_stale(root):
    marker = next(root.glob("*.json"))
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["started_at"] = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    marker.write_text(json.dumps(payload), encoding="utf-8")


def test_prompt_event_is_deduplicated_after_success(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    assert claim_hook_event("session-1", "prompt")
    finish_hook_event("session-1", "prompt", succeeded=True, selected_ids=["doc-1"])
    assert not claim_hook_event("session-1", "prompt")


def test_session_start_sources_are_distinct_and_excluded_ids_are_collected(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    assert claim_hook_event("session-1", "session-start:startup")
    finish_hook_event("session-1", "session-start:startup", succeeded=True, selected_ids=["doc-1"])
    assert claim_hook_event("session-1", "session-start:compact")
    finish_hook_event("session-1", "session-start:compact", succeeded=True, selected_ids=["doc-2"])

    assert successful_session_start_ids("session-1") == frozenset({"doc-1", "doc-2"})


def test_failed_and_stale_events_can_retry_once(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    assert claim_hook_event("session-1", "prompt")
    finish_hook_event("session-1", "prompt", succeeded=False)
    assert claim_hook_event("session-1", "prompt")
    finish_hook_event("session-1", "prompt", succeeded=False)
    assert not claim_hook_event("session-1", "prompt")

    assert claim_hook_event("session-2", "prompt")
    _make_running_marker_stale(tmp_path)
    assert claim_hook_event("session-2", "prompt")


def test_marker_files_are_owner_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    assert claim_hook_event("session-1", "prompt")
    marker = next(tmp_path.glob("*.json"))
    assert marker.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o777 == 0o700


def test_corrupt_marker_can_be_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    assert claim_hook_event("corrupt-session", "prompt")
    marker = next(tmp_path.glob("*.json"))
    marker.write_text("not-json", encoding="utf-8")
    assert claim_hook_event("corrupt-session", "prompt")


def test_attempt_token_prevents_an_old_attempt_from_finishing(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    first_attempt = claim_hook_event("token-session", "prompt")
    _make_running_marker_stale(tmp_path)
    second_attempt = claim_hook_event("token-session", "prompt")
    assert first_attempt != second_attempt

    finish_hook_event("token-session", "prompt", succeeded=True, selected_ids=["old"], attempt_token=first_attempt)
    marker = next(tmp_path.glob("*.json"))
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "running"
    finish_hook_event("token-session", "prompt", succeeded=True, selected_ids=["new"], attempt_token=second_attempt)
    assert json.loads(marker.read_text(encoding="utf-8"))["selected_ids"] == ["new"]


def test_marker_identity_fields_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path))

    session_id = "s" * 1000
    event = "e" * 1000
    assert claim_hook_event(session_id, event)
    marker = next(tmp_path.glob("*.json"))
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert len(payload["session_id"]) <= 256
    assert len(payload["event"]) <= 256
