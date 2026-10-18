import argparse
import json
import sqlite3

import pytest

import dream
import ingest
from config import load_config
from sources.codex_jsonl import CodexJSONLSource


def transcript(root, session_id):
    root.mkdir(exist_ok=True)
    path = root / f"{session_id}.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"id": session_id, "cwd": "/project"}},
        {"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": f"request for {session_id}"}],
        }},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def configured(tmp_path, sources):
    return load_config(tmp_path / "absent.toml", overrides={
        "sources": sources,
        "storage": {"db_path": str(tmp_path / "dream.db"),
                    "memory_root": str(tmp_path / "memory")},
    })


def source_config(name, kind, root):
    return {"name": name, "type": kind, "root": str(root), "enabled": True}


@pytest.mark.parametrize("broken_first", [True, False])
def test_broken_source_does_not_block_healthy_sources(tmp_path, capsys, broken_first):
    transcript(tmp_path / "codex", "healthy")
    with sqlite3.connect(tmp_path / "opencode.db") as broken:
        broken.execute("CREATE TABLE unrelated(id TEXT)")
    sources = [source_config("broken", "opencode-sqlite", tmp_path / "opencode.db"),
               source_config("healthy", "codex-jsonl", tmp_path / "codex")]
    if not broken_first:
        sources.reverse()
    conn = dream.open_db(tmp_path / "dream.db")
    result = ingest.ingest_configured_sources(conn, configured(tmp_path, sources))
    assert conn.execute("SELECT external_session_id FROM sessions").fetchall() == [("healthy",)]
    assert ("healthy", 1, 1, 1) in result
    assert "broken" in capsys.readouterr().err
    assert result.failures


def test_one_bad_session_does_not_block_other_sessions(tmp_path, monkeypatch, capsys):
    transcript(tmp_path / "codex", "bad")
    transcript(tmp_path / "codex", "healthy")

    class Source(CodexJSONLSource):
        def parse(self, ref):
            if ref.external_session_id == "bad":
                raise ValueError("invalid transcript")
            return super().parse(ref)

    monkeypatch.setattr(ingest, "create_source", lambda _kind, root: Source(root))
    conn = dream.open_db(tmp_path / "dream.db")
    result = ingest.ingest_configured_sources(conn, configured(tmp_path, [
        source_config("codex", "codex-jsonl", tmp_path / "codex"),
    ]))
    assert result == [("codex", 2, 1, 1)]
    assert conn.execute("SELECT external_session_id FROM sessions").fetchall() == [("healthy",)]
    assert "invalid transcript" in capsys.readouterr().err


def test_discovery_failure_preserves_already_ingested_sessions(tmp_path, monkeypatch):
    transcript(tmp_path / "codex", "healthy")

    class Source(CodexJSONLSource):
        def discover(self):
            yield from super().discover()
            raise OSError("source unavailable")

    monkeypatch.setattr(ingest, "create_source", lambda _kind, root: Source(root))
    conn = dream.open_db(tmp_path / "dream.db")
    result = ingest.ingest_configured_sources(conn, configured(tmp_path, [
        source_config("codex", "codex-jsonl", tmp_path / "codex"),
    ]))
    assert result == [("codex", 1, 1, 1)]
    assert result.failures
    assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1


def test_source_initialization_failure_does_not_block_next_source(tmp_path, monkeypatch):
    transcript(tmp_path / "codex", "healthy")
    create = ingest.create_source

    def sometimes_fail(kind, root):
        if kind == "opencode-sqlite":
            raise OSError("source initialization failed")
        return create(kind, root)

    monkeypatch.setattr(ingest, "create_source", sometimes_fail)
    conn = dream.open_db(tmp_path / "dream.db")
    result = ingest.ingest_configured_sources(conn, configured(tmp_path, [
        source_config("broken", "opencode-sqlite", tmp_path / "broken.db"),
        source_config("healthy", "codex-jsonl", tmp_path / "codex"),
    ]))
    assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
    assert result.failures


def test_dream_storage_error_is_not_swallowed_as_source_failure(tmp_path):
    transcript(tmp_path / "codex", "healthy")
    conn = dream.open_db(tmp_path / "dream.db")
    conn.execute("PRAGMA query_only=ON")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ingest.ingest_configured_sources(conn, configured(tmp_path, [
            source_config("codex", "codex-jsonl", tmp_path / "codex"),
        ]))


def test_transcript_removed_after_parsing_does_not_block_healthy_session(tmp_path, monkeypatch):
    transcript(tmp_path / "codex", "bad")
    transcript(tmp_path / "codex", "healthy")

    class Source(CodexJSONLSource):
        def parse(self, ref):
            parsed = super().parse(ref)
            if ref.external_session_id == "bad":
                ref.path.unlink()
            return parsed

    monkeypatch.setattr(ingest, "create_source", lambda _kind, root: Source(root))
    conn = dream.open_db(tmp_path / "dream.db")
    result = ingest.ingest_configured_sources(conn, configured(tmp_path, [
        source_config("codex", "codex-jsonl", tmp_path / "codex"),
    ]))
    assert result == [("codex", 2, 1, 1)]
    assert conn.execute("SELECT external_session_id FROM sessions").fetchall() == [("healthy",)]
    assert any("FileNotFoundError" in error for error in result.failures)


def test_keyboard_interrupt_is_not_swallowed(tmp_path, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(ingest, "create_source", interrupt)
    conn = dream.open_db(tmp_path / "dream.db")
    with pytest.raises(KeyboardInterrupt):
        ingest.ingest_configured_sources(conn, configured(tmp_path, [
            source_config("codex", "codex-jsonl", tmp_path / "codex"),
        ]))


def test_ingest_cli_reports_failure_after_processing_healthy_source(tmp_path, capsys):
    transcript(tmp_path / "codex", "healthy")
    with sqlite3.connect(tmp_path / "opencode.db") as broken:
        broken.execute("CREATE TABLE unrelated(id TEXT)")
    config = configured(tmp_path, [
        source_config("broken", "opencode-sqlite", tmp_path / "opencode.db"),
        source_config("healthy", "codex-jsonl", tmp_path / "codex"),
    ])
    args = argparse.Namespace(db=config.db_path, config=config, projects=None, source=None, force=False)
    assert dream.cmd_ingest(args) == 1
    output = capsys.readouterr()
    assert "healthy: scanned 1, ingested 1, 1 messages stored." in output.out
    assert "opencode sqlite schema missing tables" in output.err
    with sqlite3.connect(config.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
