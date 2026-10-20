"""Recall usage harvested from hook markers: evidence that never deletes a memory.

Hooks open the database read-only, so they cannot count what they deliver.
Each successful hook run leaves a lifecycle marker (``recall_hook_state``)
naming the selected document IDs and the delivered ones, those whose block
reached the agent. ``dream ingest`` harvests the markers that are
old enough that no live session still relies on them, upserts one
``recall_usage`` row per delivered document (per selected document for a
marker written before delivered IDs existed), and then deletes exactly the
markers it counted. Counting is exactly-once: harvests hold an exclusive lock
on the marker directory, and every counted marker is recorded in the
``recall_usage_markers`` ledger in the same transaction as its counts, so a
crash or a failed delete after the commit never counts a marker twice.
Interactive queries leave no markers and never count.

Usage is only evidence. With ``recall.usage_tie_break`` it breaks exact ranking
ties; it annotates the subset and audit prompts; it feeds the read-only
``dream status --usage`` report and one exporter gauge. Nothing here deletes,
hides or demotes a memory.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

from recall_documents import stable_document_id

# A marker still serves its live session: deleting a prompt marker lets
# first_prompt_only inject again, and deleting a session-start marker drops the
# prompt-path exclusion of the IDs already shown. Count a marker only a day later.
HARVEST_MIN_AGE = dt.timedelta(hours=24)
UNUSED_DAYS = 90
SINCE_KEY = "recall_usage_since"
INDEX_FILE = "MEMORY.md"
# Bounds one harvest; the remaining markers wait for the next run.
MAX_MARKERS_PER_HARVEST = 10_000
# Ledger rows outlive their markers by far: the OS temp cleaner removes old
# markers within days, and a counted marker is deleted by the same harvest.
LEDGER_RETENTION = dt.timedelta(days=30)
# recall_hook_state writes at most 256 IDs of at most 256 characters each.
_MAX_SELECTED_IDS = 256
_MAX_ID_LENGTH = 256
_MAX_MARKER_BYTES = 1 << 20

_UPSERT = """
    INSERT INTO recall_usage(document_id, first_selected_at, last_selected_at, select_count)
    VALUES (?, ?, ?, 1)
    ON CONFLICT(document_id) DO UPDATE SET
        first_selected_at = MIN(first_selected_at, excluded.first_selected_at),
        last_selected_at = MAX(last_selected_at, excluded.last_selected_at),
        select_count = select_count + 1
"""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


def _stamp(value: dt.datetime) -> str:
    """A fixed-width UTC timestamp, so MIN/MAX over the stored text are chronological."""
    return _aware(value).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _aware(parsed)


def _owned(st: os.stat_result) -> bool:
    return st.st_uid == os.getuid()


def _log(message: str) -> None:
    print(f"recall usage: {message}", file=sys.stderr)


# ------------------------------------------------------------ harvest


class _Marker(NamedTuple):
    name: str
    identity: tuple[int, int, int]  # (st_dev, st_ino, st_mtime_ns) of the file that was counted
    finished_at: str
    document_ids: tuple[str, ...]  # what the marker counts: delivered blocks, or the selection for old markers


def _open_marker_root(marker_root: Path) -> int | None:
    """A descriptor for the marker directory, or None when it is absent or unsafe.

    The directory itself must not be a symlink, must belong to this user and
    must not be writable by anyone else; every later step works relative to
    this descriptor, so the directory cannot be swapped underneath a harvest.
    The descriptor holds an exclusive lock, so two harvests (a manual
    ``dream ingest`` during the nightly run) never work on the same markers;
    closing it releases the lock. Hooks lock their own per-event files only.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(marker_root, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        _log(f"marker directory skipped ({type(exc).__name__}: not a plain directory)")
        return None
    try:
        st = os.fstat(fd)
    except OSError:
        os.close(fd)
        return None
    if not _owned(st) or (st.st_mode & 0o022) or (st.st_mode & 0o300) != 0o300:
        os.close(fd)
        _log("marker directory skipped (not owned by this user, writable by others, or not writable)")
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        _log("another harvest is running; its markers are left to it")
        return None
    except OSError as exc:
        os.close(fd)
        _log(f"marker directory skipped ({type(exc).__name__}: cannot lock it)")
        return None
    return fd


def _candidate_names(dir_fd: int) -> list[str]:
    names = []
    with os.scandir(dir_fd) as entries:
        for entry in entries:
            # Hidden names are the hook's in-flight temporary files; *.lock files
            # are its mutexes and are never touched.
            if entry.name.startswith(".") or not entry.name.endswith(".json"):
                continue
            try:
                if entry.is_file(follow_symlinks=False):
                    names.append(entry.name)
            except OSError:
                continue
    return sorted(names)


def _counted_ids(payload: dict) -> tuple[str, ...] | None:
    """The IDs a marker counts: its delivered blocks, or its selection when it predates them.

    Hooks record ``selected_ids`` (the selection, which drives the prompt-path
    exclusion) and ``delivered_ids`` (the blocks that reached the agent).
    Usage counts delivered blocks; markers written before ``delivered_ids``
    existed fall back to the selection.
    """
    values = payload["delivered_ids"] if "delivered_ids" in payload else payload.get("selected_ids")
    if not isinstance(values, list) or len(values) > _MAX_SELECTED_IDS:
        return None
    if not all(isinstance(value, str) and 0 < len(value) <= _MAX_ID_LENGTH for value in values):
        return None
    return tuple(dict.fromkeys(values))


def _read_marker(dir_fd: int, name: str, cutoff: dt.datetime) -> _Marker | None:
    """One countable marker, or None for anything else, which is then left alone.

    The file is opened without following symlinks and without blocking on a
    FIFO. Only a regular file owned by this user, of bounded size, last
    written before the cutoff, recording a succeeded session-start or prompt
    event that finished before the cutoff, with well-formed IDs, qualifies.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or not _owned(st) or st.st_size > _MAX_MARKER_BYTES:
            return None
        if st.st_mtime >= cutoff.timestamp():
            return None  # written recently: possibly still being written or in use
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(_MAX_MARKER_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > _MAX_MARKER_BYTES:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "succeeded":
        return None
    event = payload.get("event")
    if not isinstance(event, str) or not (event == "prompt" or event.startswith("session-start:")):
        return None
    finished = _parse(payload.get("finished_at"))
    if finished is None or finished >= cutoff:
        return None
    ids = _counted_ids(payload)
    if ids is None:
        return None
    return _Marker(name, (st.st_dev, st.st_ino, st.st_mtime_ns), _stamp(finished), ids)


def _record(conn: sqlite3.Connection, markers: list[_Marker], now: dt.datetime | None = None) -> list[_Marker]:
    """Count each marker not counted before, in one transaction; return the newly counted ones.

    The ledger insert and the counts commit together, so the ledger is the
    proof that a marker was counted: a marker already in it (its delete failed
    or the process died after the commit) is only deleted, never recounted, and
    its row is refreshed so pruning never forgets a marker that is still on disk.
    The ledger is keyed by marker name and finish time, because a session that
    resumes after its marker was harvested writes a new marker under the same
    name. The first counted marker opens the observation window.
    """
    now = _aware(now or _utcnow())
    counted_at = _stamp(now)
    counted: list[_Marker] = []
    with conn:
        for marker in markers:
            fresh = conn.execute(
                "INSERT OR IGNORE INTO recall_usage_markers(name, finished_at, counted_at) VALUES (?, ?, ?)",
                (marker.name, marker.finished_at, counted_at),
            ).rowcount
            if not fresh:
                # Still on disk (its delete keeps failing): keep its proof of counting alive.
                conn.execute(
                    "UPDATE recall_usage_markers SET counted_at=? WHERE name=? AND finished_at=?",
                    (counted_at, marker.name, marker.finished_at),
                )
                continue
            counted.append(marker)
            for document_id in marker.document_ids:
                conn.execute(_UPSERT, (document_id, marker.finished_at, marker.finished_at))
        if counted:
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                (SINCE_KEY, min(marker.finished_at for marker in counted)),
            )
        # After the loop, so a row refreshed above for a marker still on disk survives.
        conn.execute("DELETE FROM recall_usage_markers WHERE counted_at < ?", (_stamp(now - LEDGER_RETENTION),))
    return counted


def _delete_counted(dir_fd: int, markers: list[_Marker]) -> None:
    """Delete each counted marker, unless the file changed after it was read."""
    kept = 0
    for marker in markers:
        try:
            st = os.stat(marker.name, dir_fd=dir_fd, follow_symlinks=False)
            if (st.st_dev, st.st_ino, st.st_mtime_ns) != marker.identity:
                kept += 1
                continue
            os.unlink(marker.name, dir_fd=dir_fd)
        except FileNotFoundError:
            continue
        except OSError:
            kept += 1
    if kept:
        _log(f"{kept} counted marker(s) changed or could not be deleted and were kept")


def harvest(conn: sqlite3.Connection, marker_root, *, now: dt.datetime | None = None) -> int:
    """Count old succeeded hook markers into recall_usage; return how many were counted.

    Write path only (``dream ingest``). A marker is counted exactly once, once
    per delivered document ID, and deleted only after its counts and its ledger
    row are committed. Fresh, running, failed, malformed, foreign, symlinked or
    otherwise unexpected entries are skipped and left in place. A missing or
    locked directory counts nothing.
    """
    now = _aware(now or _utcnow())
    cutoff = now - HARVEST_MIN_AGE
    dir_fd = _open_marker_root(Path(marker_root))
    if dir_fd is None:
        return 0
    try:
        markers: list[_Marker] = []
        for name in _candidate_names(dir_fd):
            marker = _read_marker(dir_fd, name, cutoff)
            if marker is None:
                continue
            markers.append(marker)
            if len(markers) >= MAX_MARKERS_PER_HARVEST:
                break
        if not markers:
            return 0
        counted = _record(conn, markers, now)
        # Every marker read here is now in the ledger: the new ones were just
        # counted, the rest were counted by an earlier harvest.
        _delete_counted(dir_fd, markers)
        return len(counted)
    finally:
        os.close(dir_fd)


# ------------------------------------------------------------ reads (read-only safe)


def has_usage_table(conn: sqlite3.Connection) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recall_usage'"
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def usage_map(conn: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    """``{document_id: (select_count, last_selected_at)}``; empty when usage is unavailable.

    Safe on read-only connections and on databases not migrated yet.
    """
    if not has_usage_table(conn):
        return {}
    try:
        rows = conn.execute("SELECT document_id, select_count, last_selected_at FROM recall_usage").fetchall()
        return {str(document_id): (int(count), str(last)) for document_id, count, last in rows}
    except (sqlite3.Error, TypeError, ValueError):
        return {}


def usage_since(conn: sqlite3.Connection) -> str | None:
    """The start of the observation window, or None before the first counted marker."""
    if not has_usage_table(conn):
        return None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (SINCE_KEY,)).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row and row[0] else None


def memory_document_id(name: str) -> str:
    """The recall document ID of a memory file, by its path relative to the memory root.

    Consolidation passes bare filenames because ``consolidate.memory_files``
    lists top-level files only, where the name is the relative path. A nested
    file would need its path relative to the memory root (``notes/api.md``).
    """
    return stable_document_id("approved_memory", f"memory:{name}")


def annotation(usage: Mapping[str, tuple[int, str]], document_id: str) -> str:
    count, last = usage.get(document_id, (0, ""))
    return f"last recalled: {last[:10] if count and last else 'never'}; recalls: {count}"


def memory_annotations(conn: sqlite3.Connection, names: Iterable[str]) -> dict[str, str]:
    """The usage line shown under each memory file in subset and audit prompts."""
    usage = usage_map(conn)
    return {name: annotation(usage, memory_document_id(name)) for name in names}


def observation_window(conn: sqlite3.Connection) -> str:
    """How long usage has been observed, phrased for a model prompt."""
    since = usage_since(conn)
    if since:
        return since[:10]
    return "not observed yet (no recall usage has been recorded, so `never` carries no information)"


class UnusedMemory(NamedTuple):
    document_id: str
    source_path: str
    select_count: int
    last_selected_at: str | None


def unused_memory_documents(conn: sqlite3.Connection, *, days: int = UNUSED_DAYS,
                            now: dt.datetime | None = None) -> list[UnusedMemory]:
    """Approved memory documents (MEMORY.md aside) not selected in the last ``days`` days.

    Read-only. Raises sqlite3.Error when the recall tables are unreadable.
    """
    cutoff = _stamp(_aware(now or _utcnow()) - dt.timedelta(days=days))
    usage = usage_map(conn)
    rows = conn.execute(
        "SELECT id, source_path FROM recall_documents WHERE source_kind='approved_memory' "
        "ORDER BY source_path, id"
    ).fetchall()
    unused = []
    for document_id, source_path in rows:
        if source_path == INDEX_FILE:
            continue
        count, last = usage.get(document_id, (0, None))
        if not last or last < cutoff:
            unused.append(UnusedMemory(document_id, source_path, count, last or None))
    return unused


class UsageReport(NamedTuple):
    state: str  # "unavailable" (not migrated), "not-observed" (no marker counted yet) or "observed"
    since: str | None
    approved_count: int
    unused: tuple[UnusedMemory, ...]
    window_shorter_than_days: bool


def usage_report(conn: sqlite3.Connection, *, days: int = UNUSED_DAYS,
                 now: dt.datetime | None = None) -> UsageReport:
    """What ``dream status --usage`` and the exporter show. Read-only; deletes nothing."""
    now = _aware(now or _utcnow())
    if not has_usage_table(conn):
        return UsageReport("unavailable", None, 0, (), False)
    since = usage_since(conn)
    if since is None:
        return UsageReport("not-observed", None, 0, (), False)
    approved = conn.execute(
        "SELECT COUNT(*) FROM recall_documents WHERE source_kind='approved_memory' AND source_path != ?",
        (INDEX_FILE,),
    ).fetchone()[0]
    unused = tuple(unused_memory_documents(conn, days=days, now=now))
    started = _parse(since)
    shorter = started is None or now - started < dt.timedelta(days=days)
    return UsageReport("observed", since, int(approved), unused, shorter)
