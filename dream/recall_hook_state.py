"""Small, owner-only lifecycle state for sandboxed recall hooks."""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from recall_events import POLICY_VERSION


_RETRY_AFTER = dt.timedelta(seconds=10)
_MAX_SELECTED_IDS = 256
_MAX_ID_FIELD = 256


def _root() -> Path:
    configured = os.environ.get("DREAM_RECALL_STATE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"dream-recall-{os.getuid()}"


def _bounded(value: object) -> str:
    return str(value)[:_MAX_ID_FIELD]


def _stamp(value: dt.datetime | None = None) -> str:
    value = value or dt.datetime.now(dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _marker_path(session_id: str, event: str, root: Path) -> Path:
    key = f"{POLICY_VERSION}\0{session_id}\0{event}".encode("utf-8")
    return root / f"{hashlib.sha256(key).hexdigest()}.json"


def _ensure_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)


@contextlib.contextmanager
def _event_lock(path: Path):
    lock_path = path.with_suffix(".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_atomic(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def claim_hook_event(session_id: str, event: str) -> int | None:
    """Claim an event, retrying stale or failed attempts once.

    The returned attempt token must be passed to ``finish_hook_event`` so a
    stale process cannot finish a newer retry.
    """
    session_id = str(session_id)
    event = str(event)
    root = _root()
    _ensure_root(root)
    path = _marker_path(session_id, event, root)
    with _event_lock(path):
        now = dt.datetime.now(dt.timezone.utc)
        existing = _read(path) if path.exists() else None
        if path.exists() and existing is None:
            path.unlink(missing_ok=True)
        if existing:
            status = existing.get("status")
            try:
                attempts = int(existing.get("attempt_count", 1) or 1)
            except (TypeError, ValueError):
                attempts = 1
            started = _parse(existing.get("started_at"))
            stale = started is not None and now - started >= _RETRY_AFTER
            can_retry = status == "failed" and attempts < 2
            if status == "succeeded" or (status == "running" and not stale) or (status == "failed" and not can_retry):
                return None
            attempts += 1
        else:
            attempts = 1
        payload = {
            "session_id": _bounded(session_id),
            "event": _bounded(event),
            "policy_version": POLICY_VERSION,
            "status": "running",
            "attempt_count": attempts,
            "started_at": _stamp(now),
            "finished_at": None,
            "selected_ids": [],
            "error_code": None,
        }
        _write_atomic(path, payload)
        return attempts


def finish_hook_event(session_id: str, event: str, *, succeeded: bool, selected_ids: Iterable[str] = (), error_code: str | None = None, attempt_token: int | None = None) -> None:
    root = _root()
    _ensure_root(root)
    path = _marker_path(str(session_id), str(event), root)
    with _event_lock(path):
        payload = _read(path)
        if payload is None:
            return
        try:
            current_attempt = int(payload.get("attempt_count", 0))
        except (TypeError, ValueError):
            return
        if attempt_token is not None and current_attempt != attempt_token:
            return
        bounded_ids = [_bounded(value) for value in list(selected_ids)[:_MAX_SELECTED_IDS]] if succeeded else []
        payload.update({
            "status": "succeeded" if succeeded else "failed",
            "finished_at": _stamp(),
            "selected_ids": bounded_ids,
            "error_code": _bounded(error_code) if error_code else None,
        })
        _write_atomic(path, payload)


def successful_session_start_ids(session_id: str) -> frozenset[str]:
    root = _root()
    if not root.is_dir():
        return frozenset()
    session_id = _bounded(session_id)
    selected: set[str] = set()
    for path in root.glob("*.json"):
        payload = _read(path)
        if not payload or payload.get("session_id") != session_id:
            continue
        if payload.get("status") != "succeeded" or not str(payload.get("event", "")).startswith("session-start:"):
            continue
        values = payload.get("selected_ids")
        if isinstance(values, list):
            selected.update(_bounded(value) for value in values[:_MAX_SELECTED_IDS])
    return frozenset(selected)
