import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import consolidate as C


SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _db():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def _distilled(conn, sid, ended, key, note="x"):
    conn.execute(
        "INSERT INTO sessions(session_id, source, external_session_id, project_slug, jsonl_path, "
        "jsonl_mtime, ended_at, total_chars) VALUES (?, 'codex', ?, 'p', '/p', 1, ?, 1000)",
        (sid, sid, ended),
    )
    conn.execute(
        "INSERT INTO distilled(session_id, distillation_key, notes_json) VALUES (?, ?, ?)",
        (sid, key, json.dumps({"facts": [note]})),
    )
    conn.commit()


def test_historical_session_is_selected_without_timestamp_cursor():
    conn = _db()
    _distilled(conn, "old", "2020-01-01T00:00:00Z", "key-old")
    assert [row[0] for row in C.collect_distilled_batch(conn, None)] == ["old"]


def test_success_consumes_only_rows_that_fit_batch(monkeypatch, tmp_path):
    conn = _db()
    _distilled(conn, "a", "2020-01-01T00:00:00Z", "key-a", "a" * 80)
    _distilled(conn, "b", "2020-01-02T00:00:00Z", "key-b", "b" * 80)
    monkeypatch.setattr(C, "MAX_DISTILLED_BATCH_CHARS", 180)
    monkeypatch.setattr(
        C,
        "generate",
        lambda *a, **k: SimpleNamespace(output={"suggestions": []}, provider="fake", model="fake"),
    )

    C.consolidate(conn, tmp_path)
    consumed = {
        row[0] for row in conn.execute("SELECT distillation_key FROM consolidated_distillations")
    }
    assert consumed == {"key-a"}
    assert [row[0] for row in C.collect_distilled_batch(conn, None)] == ["b"]


def test_provider_failure_consumes_nothing(monkeypatch, tmp_path):
    conn = _db()
    _distilled(conn, "a", "2020-01-01T00:00:00Z", "key-a")

    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(C, "generate", fail)
    with pytest.raises(RuntimeError, match="provider failed"):
        C.consolidate(conn, tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM consolidated_distillations").fetchone()[0] == 0
