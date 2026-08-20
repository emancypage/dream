import io
import json
import sys

from recall_context import hook_success_json, parse_hook_payload, run_context
from recall_types import AdapterSettings, RecallSettings


def _settings():
    return RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False))


def test_parse_hook_payload_and_exact_success_json():
    query = parse_hook_payload("UserPromptSubmit", {"session_id": "s", "prompt": "hello", "cwd": "/tmp", "repository_roots": ["/tmp"]})
    assert query.hook_event == "prompt"
    assert json.loads(hook_success_json("ctx")) == {"continue": True, "hookSpecificOutput": {"additionalContext": "ctx"}}


def test_missing_hook_fields_fail_open_parser():
    import pytest
    with pytest.raises(ValueError):
        parse_hook_payload("SessionStart", {"cwd": "/tmp"})


def test_context_explain_has_bounded_diagnostics(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    result = run_context(conn, parse_hook_payload("UserPromptSubmit", {"session_id": "s", "prompt": "missing", "cwd": "/tmp"}), _settings(), explain=True)
    assert result.diagnostics.selected_count == 0


def test_readonly_database_opener_works_without_parent_write_access(tmp_path):
    from dream import open_db, open_db_readonly

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = open_db(db_path)
    conn.close()
    db_dir.chmod(0o555)
    try:
        readonly = open_db_readonly(db_path)
        assert readonly.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        readonly.close()
        assert sorted(path.name for path in db_dir.iterdir()) == ["dream.db"]
    finally:
        db_dir.chmod(0o755)


def test_status_reads_database_without_parent_write_access(tmp_path, capsys):
    from types import SimpleNamespace

    from config import load_config
    from dream import cmd_status, open_db

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = open_db(db_path)
    conn.close()
    db_dir.chmod(0o555)
    try:
        rc = cmd_status(SimpleNamespace(db=db_path, config=load_config(tmp_path / "config.toml")))
        assert rc == 0
        assert "Sessions ingested:  0" in capsys.readouterr().out
    finally:
        db_dir.chmod(0o755)


def test_readonly_database_opener_fails_closed_while_wal_is_present(tmp_path):
    import sqlite3

    from dream import open_db, open_db_readonly

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = open_db(db_path)
    conn.close()
    writer = sqlite3.connect(db_path)
    writer.execute("PRAGMA wal_autocheckpoint = 0")
    writer.execute("INSERT INTO meta(key, value) VALUES('wal-test', '1')")
    writer.commit()
    try:
        assert db_path.with_name("dream.db-wal").exists()
        import pytest
        with pytest.raises(sqlite3.OperationalError, match="WAL"):
            open_db_readonly(db_path)
    finally:
        writer.close()
