"""Tests that consolidate() still runs a staleness-only pass when there is no new
session batch, as long as the memory store isn't empty — and that doing so never
moves the incremental watermark (nothing new was actually consolidated)."""
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import consolidate as C  # noqa: E402

SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def test_calls_model_with_empty_batch_when_memory_nonempty(monkeypatch):
    calls = []

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append(prompt)
        return SimpleNamespace(output={"suggestions": []}, provider="fake", model="fake")

    monkeypatch.setattr(C, "generate", fake_generate)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        conn = _db()
        result = C.consolidate(conn, root, since_iso="2026-01-01T00:00:00")

    assert len(calls) == 1
    assert result == []


def test_skips_model_when_batch_and_memory_both_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(C, "generate", lambda *a, **k: calls.append(1))

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)  # empty — no *.md files
        conn = _db()
        result = C.consolidate(conn, root, since_iso="2026-01-01T00:00:00")

    assert calls == []
    assert result == []


def test_empty_batch_does_not_move_watermark(monkeypatch):
    monkeypatch.setattr(
        C, "generate",
        lambda *a, **k: SimpleNamespace(output={"suggestions": []}, provider="fake", model="fake"),
    )

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        conn = _db()
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('last_consolidate_at', '2026-06-01T00:00:00')"
        )
        conn.commit()
        C.consolidate(conn, root, since_iso="2026-01-01T00:00:00")
        row = conn.execute("SELECT value FROM meta WHERE key='last_consolidate_at'").fetchone()

    assert row[0] == "2026-06-01T00:00:00"
