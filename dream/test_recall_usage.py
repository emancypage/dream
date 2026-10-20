"""Recall usage: harvesting hook markers, the read-only report and fail-open reads.

Usage is evidence for people and for the audit prompt. It never deletes or
hides a memory: the only files harvest removes are hook markers it counted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import io
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import dream
import recall_usage
from config import ConfigError, load_config
from recall_context import run_context
from recall_documents import stable_document_id, synchronize_recall_documents
from recall_hook_state import claim_hook_event, finish_hook_event, marker_root
from recall_types import RecallQuery
from storage import open_db_readonly


NOW = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)
OLD = NOW - dt.timedelta(hours=30)


def _stamp(value: dt.datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _root(tmp_path) -> Path:
    root = tmp_path / "markers"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def _marker(root: Path, name: str, *, ids=("doc-a",), event="session-start:startup", status="succeeded",
            finished: dt.datetime | None = OLD, mtime: dt.datetime | None = None, payload=None) -> Path:
    """Write one marker in the recall_hook_state format, dated like the hook would date it."""
    if payload is None:
        payload = {
            "session_id": name,
            "event": event,
            "policy_version": "recall-v1",
            "status": status,
            "attempt_count": 1,
            "started_at": _stamp((finished or NOW) - dt.timedelta(seconds=1)),
            "finished_at": _stamp(finished) if finished else None,
            "selected_ids": list(ids),
            "error_code": None,
        }
    path = root / f"{name}.json"
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    stamp = (mtime or finished or NOW).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _db(tmp_path):
    return dream.open_db(tmp_path / "dream.db")


def _usage(conn) -> dict:
    return {row[0]: row[1:] for row in conn.execute(
        "SELECT document_id, select_count, first_selected_at, last_selected_at FROM recall_usage ORDER BY document_id")}


def _since(conn):
    row = conn.execute("SELECT value FROM meta WHERE key='recall_usage_since'").fetchone()
    return row[0] if row else None


# ------------------------------------------------------------ harvest


def test_harvest_counts_old_succeeded_markers_and_deletes_only_those(tmp_path):
    root = _root(tmp_path)
    start = _marker(root, "start", ids=("doc-a", "doc-b"))
    prompt = _marker(root, "prompt", ids=("doc-a",), event="prompt", finished=OLD + dt.timedelta(minutes=5))
    failed = _marker(root, "failed", status="failed", ids=())
    running = _marker(root, "running", status="running", finished=None, mtime=OLD)
    lock = root / "start.lock"
    lock.write_text("", encoding="utf-8")
    conn = _db(tmp_path)

    assert recall_usage.harvest(conn, root, now=NOW) == 2

    assert _usage(conn) == {
        "doc-a": (2, "2026-10-19T06:00:00Z", "2026-10-19T06:05:00Z"),
        "doc-b": (1, "2026-10-19T06:00:00Z", "2026-10-19T06:00:00Z"),
    }
    assert not start.exists() and not prompt.exists()
    assert failed.exists() and running.exists() and lock.exists()


def test_harvest_is_idempotent(tmp_path):
    root = _root(tmp_path)
    _marker(root, "one", ids=("doc-a",))
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    first = _usage(conn)
    assert recall_usage.harvest(conn, root, now=NOW) == 0
    assert recall_usage.harvest(conn, root, now=NOW + dt.timedelta(days=3)) == 0
    assert _usage(conn) == first == {"doc-a": (1, "2026-10-19T06:00:00Z", "2026-10-19T06:00:00Z")}


def test_harvest_counts_a_document_once_per_marker_and_keeps_the_selection_window(tmp_path):
    root = _root(tmp_path)
    _marker(root, "late", ids=("doc-a", "doc-a", "doc-a"), finished=NOW - dt.timedelta(hours=25))
    _marker(root, "early", ids=("doc-a",), finished=NOW - dt.timedelta(days=9))
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, root, now=NOW) == 2
    assert _usage(conn) == {"doc-a": (2, "2026-10-11T12:00:00Z", "2026-10-19T11:00:00Z")}
    assert recall_usage.usage_map(conn) == {"doc-a": (2, "2026-10-19T11:00:00Z")}


def test_markers_younger_than_24_hours_are_left_for_their_live_session(tmp_path):
    root = _root(tmp_path)
    fresh = _marker(root, "fresh", ids=("doc-a",), finished=NOW - dt.timedelta(hours=23, minutes=59))
    # Old finished_at but just rewritten: possibly still being written; wait for it too.
    touched = _marker(root, "touched", ids=("doc-b",), finished=OLD, mtime=NOW - dt.timedelta(minutes=1))
    conn = _db(tmp_path)

    assert recall_usage.harvest(conn, root, now=NOW) == 0
    assert _usage(conn) == {}
    assert fresh.exists() and touched.exists()

    assert recall_usage.harvest(conn, root, now=NOW + dt.timedelta(hours=25)) == 2
    assert set(_usage(conn)) == {"doc-a", "doc-b"}
    assert not fresh.exists() and not touched.exists()


@pytest.mark.parametrize("payload", [
    "{not json",
    "[]",
    json.dumps({"status": "succeeded", "event": "prompt", "finished_at": "2026-10-01T00:00:00Z",
                "selected_ids": "doc-a"}),
    json.dumps({"status": "succeeded", "event": "prompt", "finished_at": "yesterday", "selected_ids": ["doc-a"]}),
    json.dumps({"status": "succeeded", "event": "prompt", "selected_ids": ["doc-a"]}),
    json.dumps({"status": "succeeded", "event": "character", "finished_at": "2026-10-01T00:00:00Z",
                "selected_ids": ["doc-a"]}),
    json.dumps({"status": "succeeded", "event": "prompt", "finished_at": "2026-10-01T00:00:00Z",
                "selected_ids": [7, None, {"id": "doc-a"}]}),
    b"\xff\xfe{}",
])
def test_malformed_markers_are_skipped_and_kept(tmp_path, payload):
    root = _root(tmp_path)
    bad = _marker(root, "bad", payload=payload)
    good = _marker(root, "good", ids=("doc-good",))
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert set(_usage(conn)) == {"doc-good"}
    assert bad.exists() and not good.exists()


def test_symlinks_directories_and_hidden_files_are_never_followed_or_deleted(tmp_path):
    root = _root(tmp_path)
    elsewhere = _root(tmp_path / "elsewhere")
    target = _marker(elsewhere, "target", ids=("doc-outside",))
    link = root / "link.json"
    link.symlink_to(target)
    (root / "directory.json").mkdir()
    hidden = _marker(root, ".hidden", ids=("doc-hidden",))
    fifo = root / "pipe.json"
    os.mkfifo(fifo)
    conn = _db(tmp_path)

    assert recall_usage.harvest(conn, root, now=NOW) == 0
    assert _usage(conn) == {}
    assert link.is_symlink() and target.exists() and hidden.exists()
    assert (root / "directory.json").is_dir() and fifo.exists()


def test_foreign_markers_are_skipped_and_kept(tmp_path, monkeypatch):
    root = _root(tmp_path)
    foreign = _marker(root, "foreign", ids=("doc-foreign",))
    own = _marker(root, "own", ids=("doc-own",))
    foreign_inode = foreign.stat().st_ino
    monkeypatch.setattr(recall_usage, "_owned", lambda st: st.st_ino != foreign_inode)
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert set(_usage(conn)) == {"doc-own"}
    assert foreign.exists() and not own.exists()


def test_an_unsafe_marker_root_is_not_harvested(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    marker = _marker(root, "one")
    conn = _db(tmp_path)

    link = tmp_path / "linked-root"
    link.symlink_to(root, target_is_directory=True)
    assert recall_usage.harvest(conn, link, now=NOW) == 0

    root.chmod(0o777)
    assert recall_usage.harvest(conn, root, now=NOW) == 0
    root.chmod(0o700)

    monkeypatch.setattr(recall_usage, "_owned", lambda st: False)
    assert recall_usage.harvest(conn, root, now=NOW) == 0

    assert marker.exists() and _usage(conn) == {}
    assert "recall usage" in capsys.readouterr().err


def test_a_missing_marker_root_is_a_quiet_no_op_and_is_not_created(tmp_path, capsys):
    conn = _db(tmp_path)
    missing = tmp_path / "no-markers"
    assert recall_usage.harvest(conn, missing, now=NOW) == 0
    assert not missing.exists()
    assert capsys.readouterr().err == ""


def test_the_first_counted_marker_starts_the_observation_window(tmp_path):
    # Hooks are not installed by default: harvesting an empty or missing marker
    # directory observes nothing, so it must not start a window in which every
    # file would look unused.
    root = _root(tmp_path)
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, tmp_path / "no-markers", now=NOW) == 0
    assert recall_usage.harvest(conn, root, now=NOW) == 0
    _marker(root, "fresh", finished=NOW - dt.timedelta(hours=1))
    assert recall_usage.harvest(conn, root, now=NOW) == 0
    assert _since(conn) is None and recall_usage.usage_since(conn) is None

    _marker(root, "later", ids=(), event="prompt", finished=OLD + dt.timedelta(hours=2))
    _marker(root, "earlier", finished=OLD)
    assert recall_usage.harvest(conn, root, now=NOW) == 2
    assert _since(conn) == recall_usage.usage_since(conn) == "2026-10-19T06:00:00Z"

    _marker(root, "older", finished=OLD - dt.timedelta(days=5))
    assert recall_usage.harvest(conn, root, now=NOW + dt.timedelta(days=2)) == 2
    assert _since(conn) == "2026-10-19T06:00:00Z"


def test_a_failed_write_deletes_no_marker(tmp_path):
    root = _root(tmp_path)
    marker = _marker(root, "one")
    conn = _db(tmp_path)
    conn.execute("DROP TABLE recall_usage")
    with pytest.raises(sqlite3.Error):
        recall_usage.harvest(conn, root, now=NOW)
    assert marker.exists()


def _ledger(conn):
    return conn.execute("SELECT name, finished_at, counted_at FROM recall_usage_markers ORDER BY name").fetchall()


def test_counted_markers_are_recorded_in_the_ledger_with_their_counts(tmp_path):
    root = _root(tmp_path)
    _marker(root, "one", ids=("doc-a",))
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert _ledger(conn) == [("one.json", "2026-10-19T06:00:00Z", "2026-10-20T12:00:00Z")]


def test_a_crash_after_the_commit_never_counts_a_marker_twice(tmp_path, monkeypatch):
    root = _root(tmp_path)
    marker = _marker(root, "one", ids=("doc-a",))
    conn = _db(tmp_path)

    def killed(*args):
        raise KeyboardInterrupt("killed after the commit")

    monkeypatch.setattr(recall_usage, "_delete_counted", killed)
    with pytest.raises(KeyboardInterrupt):
        recall_usage.harvest(conn, root, now=NOW)
    assert marker.exists()
    monkeypatch.undo()

    assert recall_usage.harvest(conn, root, now=NOW) == 0
    assert _usage(conn) == {"doc-a": (1, "2026-10-19T06:00:00Z", "2026-10-19T06:00:00Z")}
    assert not marker.exists()


def test_a_failed_delete_never_counts_a_marker_twice(tmp_path, monkeypatch):
    root = _root(tmp_path)
    marker = _marker(root, "one", ids=("doc-a",))
    conn = _db(tmp_path)
    monkeypatch.setattr(recall_usage, "_delete_counted", lambda dir_fd, markers: None)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert recall_usage.harvest(conn, root, now=NOW + dt.timedelta(days=1)) == 0
    assert marker.exists()
    monkeypatch.undo()

    assert recall_usage.harvest(conn, root, now=NOW + dt.timedelta(days=2)) == 0
    assert _usage(conn) == {"doc-a": (1, "2026-10-19T06:00:00Z", "2026-10-19T06:00:00Z")}
    assert not marker.exists()


def _interleave(monkeypatch, second):
    """Run ``second()`` while the outer harvest holds its marker list, before it writes."""
    real = recall_usage._record
    results = []

    def record(conn, markers, now=None):
        monkeypatch.setattr(recall_usage, "_record", real)
        results.append(second())
        return real(conn, markers, now)

    monkeypatch.setattr(recall_usage, "_record", record)
    return results


def test_a_concurrent_harvest_finds_the_directory_locked(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    _marker(root, "one", ids=("doc-a",))
    conn = _db(tmp_path)
    other = dream.open_db(tmp_path / "dream.db")
    inner = _interleave(monkeypatch, lambda: recall_usage.harvest(other, root, now=NOW))
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert inner == [0]
    assert "another harvest is running" in capsys.readouterr().err
    assert {doc: row[0] for doc, row in _usage(conn).items()} == {"doc-a": 1}


def test_the_ledger_alone_keeps_interleaved_harvests_exactly_once(tmp_path, monkeypatch):
    # Even if the directory lock were bypassed, the ledger insert decides who counts.
    root = _root(tmp_path)
    _marker(root, "one", ids=("doc-a",))
    _marker(root, "two", ids=("doc-b",))
    conn = _db(tmp_path)
    other = dream.open_db(tmp_path / "dream.db")
    monkeypatch.setattr(recall_usage.fcntl, "flock", lambda fd, operation: None)
    inner = _interleave(monkeypatch, lambda: recall_usage.harvest(other, root, now=NOW))
    assert recall_usage.harvest(conn, root, now=NOW) == 0
    assert inner == [2]
    assert {doc: row[0] for doc, row in _usage(conn).items()} == {"doc-a": 1, "doc-b": 1}
    assert list(root.glob("*.json")) == []


def test_a_held_directory_lock_defers_the_harvest(tmp_path, capsys):
    root = _root(tmp_path)
    marker = _marker(root, "one", ids=("doc-a",))
    conn = _db(tmp_path)
    held = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert recall_usage.harvest(conn, root, now=NOW) == 0
        assert marker.exists() and _usage(conn) == {}
        assert "another harvest is running" in capsys.readouterr().err
    finally:
        os.close(held)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert not marker.exists()


def test_a_resumed_session_marker_under_the_same_name_is_counted_again(tmp_path):
    root = _root(tmp_path)
    _marker(root, "session", ids=("doc-a",), finished=OLD)
    conn = _db(tmp_path)
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    _marker(root, "session", ids=("doc-a",), finished=NOW + dt.timedelta(days=1))
    assert recall_usage.harvest(conn, root, now=NOW + dt.timedelta(days=3)) == 1
    assert _usage(conn)["doc-a"][0] == 2


def test_ledger_rows_older_than_30_days_are_pruned(tmp_path):
    root = _root(tmp_path)
    conn = _db(tmp_path)
    conn.executemany("INSERT INTO recall_usage_markers VALUES (?, ?, ?)", [
        ("stale.json", "2026-09-01T00:00:00Z", _stamp(NOW - dt.timedelta(days=31))),
        ("recent.json", "2026-09-25T00:00:00Z", _stamp(NOW - dt.timedelta(days=29))),
    ])
    conn.commit()
    _marker(root, "one")
    assert recall_usage.harvest(conn, root, now=NOW) == 1
    assert [row[0] for row in _ledger(conn)] == ["one.json", "recent.json"]


def test_markers_written_by_the_hook_state_module_are_harvested(tmp_path, monkeypatch):
    root = tmp_path / "hook-state"
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(root))
    token = claim_hook_event("session-1", "session-start:startup")
    finish_hook_event("session-1", "session-start:startup", succeeded=True, selected_ids=["doc-a", "doc-b"],
                      attempt_token=token)
    token = claim_hook_event("session-1", "prompt")
    finish_hook_event("session-1", "prompt", succeeded=True, selected_ids=["doc-b"], attempt_token=token)
    conn = _db(tmp_path)
    now = dt.datetime.now(dt.timezone.utc)

    assert marker_root() == root
    assert recall_usage.harvest(conn, root, now=now) == 0
    assert recall_usage.harvest(conn, root, now=now + dt.timedelta(hours=25)) == 2
    assert {doc: row[0] for doc, row in _usage(conn).items()} == {"doc-a": 1, "doc-b": 2}
    assert list(root.glob("*.json")) == []
    assert len(list(root.glob("*.lock"))) == 2


# ------------------------------------------------------------ dream ingest


def _ingest_args(tmp_path):
    config = load_config(tmp_path / "missing-config.toml", overrides={
        "storage": {"db_path": str(tmp_path / "dream.db"), "memory_root": str(tmp_path / "memory")},
    })
    return argparse.Namespace(db=tmp_path / "dream.db", config=config, projects=None, source=None, force=False)


def test_dream_ingest_harvests_markers_before_ingesting(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(root))
    _marker(root, "old", ids=("doc-a",), finished=dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=2))
    order = []
    real_harvest = recall_usage.harvest

    def harvest(conn, marker_root, **kwargs):
        order.append("harvest")
        return real_harvest(conn, marker_root, **kwargs)

    def sources(conn, config=None, selected=None, force=False):
        order.append("ingest")
        return [("codex", 0, 0, 0)]

    monkeypatch.setattr(recall_usage, "harvest", harvest)
    monkeypatch.setattr(dream, "ingest_configured_sources", sources)
    assert dream.cmd_ingest(_ingest_args(tmp_path)) == 0
    assert order == ["harvest", "ingest"]
    assert "recall usage: counted 1 hook marker" in capsys.readouterr().out
    with sqlite3.connect(tmp_path / "dream.db") as conn:
        assert conn.execute("SELECT document_id, select_count FROM recall_usage").fetchall() == [("doc-a", 1)]


def test_a_harvest_failure_is_logged_and_ingest_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(_root(tmp_path)))

    def boom(conn, marker_root, **kwargs):
        raise OSError("marker directory exploded")

    monkeypatch.setattr(recall_usage, "harvest", boom)
    monkeypatch.setattr(dream, "ingest_configured_sources",
                        lambda conn, config=None, selected=None, force=False: [("codex", 1, 1, 3)])
    assert dream.cmd_ingest(_ingest_args(tmp_path)) == 0
    captured = capsys.readouterr()
    assert "codex: scanned 1, ingested 1, 3 messages stored." in captured.out
    assert "recall usage harvest skipped: OSError: marker directory exploded" in captured.err


# ------------------------------------------------------------ reads, migration, config


def _unmigrated(tmp_path) -> Path:
    path = tmp_path / "legacy.db"
    conn = dream.open_db(path)
    conn.execute("DROP TABLE recall_usage")
    conn.execute("DROP TABLE recall_usage_markers")
    conn.commit()
    conn.close()
    for backup in tmp_path.glob("legacy.db.bak-*"):
        backup.unlink()
    return path


def test_usage_reads_fail_open_on_a_read_only_unmigrated_database(tmp_path):
    conn = open_db_readonly(_unmigrated(tmp_path))
    assert recall_usage.usage_map(conn) == {}
    assert recall_usage.usage_since(conn) is None
    assert recall_usage.memory_annotations(conn, ["topic.md"]) == {"topic.md": "last recalled: never; recalls: 0"}
    assert recall_usage.usage_map(sqlite3.connect(":memory:")) == {}


def test_hook_recall_with_the_tie_break_on_works_on_an_unmigrated_database(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "api.md").write_text("Postgres migration decision.", encoding="utf-8")
    path = _unmigrated(tmp_path)
    conn = sqlite3.connect(path)
    synchronize_recall_documents(conn, memory, include_raw_transcripts=False, constitution_path=None)
    conn.close()
    settings = load_config(tmp_path / "missing.toml", overrides={"recall": {"usage_tie_break": True}}).recall_settings()
    assert settings.usage_tie_break is True
    readonly = open_db_readonly(path)
    query = RecallQuery("postgres migration", "s", "prompt", str(memory), (), 4000, frozenset(), False)
    result = run_context(readonly, query, settings, allow_cache_writes=False, memory_root=memory, constitution_path=None)
    assert [candidate.document.source_path for candidate in result.selected] == ["api.md"]
    assert not list(tmp_path.glob("legacy.db-*"))


def test_hook_context_with_the_tie_break_on_reads_a_read_only_unmigrated_database(tmp_path, monkeypatch, capsys):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "api.md").write_text("Postgres migration decision.", encoding="utf-8")
    store = tmp_path / "store"
    store.mkdir()
    path = _unmigrated(tmp_path)
    conn = sqlite3.connect(path)
    synchronize_recall_documents(conn, memory, include_raw_transcripts=False, constitution_path=None)
    conn.close()
    path = path.rename(store / "dream.db")
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[storage]\nmemory_root = "{memory}"\n\n[recall]\nusage_tie_break = true\n',
                           encoding="utf-8")
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(tmp_path / "hook-state"))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"session_id": "s", "prompt": "Postgres migration", "cwd": str(memory)})))
    store.chmod(0o555)
    try:
        rc = dream.cmd_context(SimpleNamespace(context_cmd="prompt", explain=False,
                                               config=load_config(config_path), db=path))
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert "Postgres migration decision." in output["hookSpecificOutput"]["additionalContext"]
        assert not list(store.glob("dream.db-*"))
    finally:
        store.chmod(0o755)


@pytest.mark.parametrize("table, columns", [
    ("recall_usage", {"document_id", "first_selected_at", "last_selected_at", "select_count"}),
    ("recall_usage_markers", {"name", "finished_at", "counted_at"}),
])
def test_migration_backs_up_once_then_adds_the_usage_tables(tmp_path, table, columns):
    db_path = tmp_path / "dream.db"
    conn = dream.open_db(db_path)
    conn.execute(f"DROP TABLE {table}")
    conn.commit()
    conn.close()
    assert not list(tmp_path.glob("dream.db.bak-*"))

    conn = dream.open_db(db_path)
    assert {row[1] for row in conn.execute(f"PRAGMA table_info({table})")} == columns
    assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 4
    conn.close()
    [backup] = list(tmp_path.glob("dream.db.bak-provider-migration-*"))
    legacy = sqlite3.connect(backup)
    try:
        assert legacy.execute("SELECT name FROM sqlite_master WHERE name=?", (table,)).fetchone() is None
    finally:
        legacy.close()
    dream.open_db(db_path).close()
    assert list(tmp_path.glob("dream.db.bak-provider-migration-*")) == [backup]


def test_usage_tie_break_is_off_by_default_and_must_be_a_boolean(tmp_path):
    assert load_config(tmp_path / "missing.toml").recall_settings().usage_tie_break is False
    assert load_config(Path("/nonexistent/dream.toml")).data["recall"]["usage_tie_break"] is False
    with pytest.raises(ConfigError, match="recall.usage_tie_break must be a boolean"):
        load_config(tmp_path / "missing.toml", overrides={"recall": {"usage_tie_break": "yes"}})


# ------------------------------------------------------------ status --usage


def _usage_store(tmp_path, *, harvested=True):
    root = tmp_path / "memory"
    root.mkdir()
    files = {
        "MEMORY.md": "- [Fresh](fresh.md) — used every day\n- [Old](old.md) — last touched long ago\n",
        "fresh.md": "Fresh facts.\n",
        "old.md": "Old facts.\n",
        "lapsed.md": "Facts recalled once, long ago.\n",
    }
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")
    db_path = tmp_path / "dream.db"
    conn = dream.open_db(db_path)
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, constitution_path=None)
    rows = [
        (stable_document_id("approved_memory", "memory:fresh.md"), "2026-07-01T00:00:00Z", "2026-10-18T09:00:00Z", 12),
        (stable_document_id("approved_memory", "memory:lapsed.md"), "2026-07-02T00:00:00Z", "2026-07-02T00:00:00Z", 1),
    ]
    conn.executemany("INSERT INTO recall_usage VALUES (?, ?, ?, ?)", rows)
    if harvested:
        conn.execute("INSERT INTO meta(key, value) VALUES ('recall_usage_since', '2026-06-30T22:00:00Z')")
    conn.commit()
    conn.close()
    config = load_config(tmp_path / "missing.toml", overrides={
        "storage": {"db_path": str(db_path), "memory_root": str(root)}})
    return root, db_path, config


def _status(db_path, config, usage=True):
    return dream.cmd_status(argparse.Namespace(db=db_path, config=config, usage=usage))


def test_unused_memory_documents_cover_never_and_long_ago(tmp_path):
    _root_dir, db_path, _config = _usage_store(tmp_path)
    conn = open_db_readonly(db_path)
    unused = recall_usage.unused_memory_documents(conn, now=NOW)
    assert [(row.source_path, row.select_count, row.last_selected_at) for row in unused] == [
        ("lapsed.md", 1, "2026-07-02T00:00:00Z"),
        ("old.md", 0, None),
    ]
    assert [row.source_path for row in recall_usage.unused_memory_documents(conn, now=NOW, days=200)] == ["old.md"]


def test_status_usage_lists_unrecalled_files_with_their_index_lines(tmp_path, monkeypatch, capsys):
    root, db_path, config = _usage_store(tmp_path)
    monkeypatch.setattr(recall_usage, "_utcnow", lambda: NOW)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    db_before = db_path.read_bytes()

    assert _status(db_path, config) == 0

    out = capsys.readouterr().out
    report = out.split("Recall usage", 1)[1]
    assert "observed since 2026-06-30" in report
    assert "2 of 3 approved memory files not recalled in the last 90 days" in report
    assert "old.md (last recalled: never; recalls: 0)" in report
    assert "- [Old](old.md) — last touched long ago" in report
    assert "lapsed.md (last recalled: 2026-07-02; recalls: 1)" in report
    assert "(not in MEMORY.md)" in report
    assert "fresh.md" not in report
    assert "nothing is deleted" in report
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert db_path.read_bytes() == db_before
    assert not list(tmp_path.glob("dream.db-*"))


def test_status_usage_says_when_the_window_is_shorter_than_90_days(tmp_path, monkeypatch, capsys):
    _root_dir, db_path, config = _usage_store(tmp_path)
    monkeypatch.setattr(recall_usage, "_utcnow", lambda: dt.datetime(2026, 7, 31, tzinfo=dt.timezone.utc))
    assert _status(db_path, config) == 0
    assert "shorter than 90 days" in capsys.readouterr().out


def test_status_usage_before_the_first_harvest_lists_nothing(tmp_path, capsys):
    _root_dir, db_path, config = _usage_store(tmp_path, harvested=False)
    assert _status(db_path, config) == 0
    report = capsys.readouterr().out.split("Recall usage", 1)[1]
    assert "not observed yet" in report
    assert "old.md" not in report


def test_status_usage_on_an_unmigrated_database_reports_it_and_changes_nothing(tmp_path, capsys):
    path = _unmigrated(tmp_path)
    config = load_config(tmp_path / "missing.toml", overrides={
        "storage": {"db_path": str(path), "memory_root": str(tmp_path / "memory")}})
    assert _status(path, config) == 0
    assert "not available" in capsys.readouterr().out
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='recall_usage'").fetchone() is None
    finally:
        conn.close()


def test_status_without_usage_flag_prints_no_usage_report(tmp_path, capsys):
    _root_dir, db_path, config = _usage_store(tmp_path)
    assert _status(db_path, config, usage=False) == 0
    assert "Recall usage" not in capsys.readouterr().out


def test_status_usage_flag_is_wired_into_the_cli(tmp_path, monkeypatch, capsys):
    root, db_path, _config = _usage_store(tmp_path)
    monkeypatch.setattr(recall_usage, "_utcnow", lambda: NOW)
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[storage]\ndb_path = "{db_path}"\nmemory_root = "{root}"\n', encoding="utf-8")
    assert dream.main(["--config", str(config_path), "status", "--usage"]) == 0
    assert "old.md (last recalled: never; recalls: 0)" in capsys.readouterr().out


def test_status_usage_with_unreadable_recall_tables_reports_it(tmp_path, capsys):
    _root_dir, db_path, config = _usage_store(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE recall_documents")
    conn.commit()
    conn.close()
    assert _status(db_path, config) == 0
    assert "not available (OperationalError" in capsys.readouterr().out
