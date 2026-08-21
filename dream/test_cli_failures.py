import argparse
import sqlite3

import dream
from config import load_config


def test_distill_returns_nonzero_when_provider_call_fails(monkeypatch, tmp_path):
    conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(dream, "open_db", lambda _path: conn)
    monkeypatch.setattr(dream, "sessions_needing_distill", lambda *a, **k: [("session", 1000)])

    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(dream, "distill_session", fail)
    args = argparse.Namespace(
        db=tmp_path / "dream.db",
        min_chars=500,
        project=None,
        config=load_config(tmp_path / "absent.toml"),
        limit=1,
        max_sessions=5,
        yes=True,
        model=None,
        refresh=False,
    )
    assert dream.cmd_distill(args) == 1


def test_read_only_commands_work_without_database_directory_write_access(tmp_path, capsys):
    from types import SimpleNamespace

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = dream.open_db(db_path)
    conn.close()
    db_dir.chmod(0o555)
    try:
        config = load_config(tmp_path / "absent.toml")
        assert dream.cmd_estimate(SimpleNamespace(
            db=db_path, min_chars=500, project=None, config=config, refresh=False, verbose=False,
        )) == 0
        assert dream.cmd_search(SimpleNamespace(
            db=db_path, query="missing", limit=20, role=None, project=None,
        )) == 0
        assert dream.cmd_suggestions(SimpleNamespace(
            db=db_path, memory=str(tmp_path / "memory"), suggestions_cmd="list", config=config,
        )) == 0
        assert '"suggestions": []' in capsys.readouterr().out
    finally:
        db_dir.chmod(0o755)


def test_curate_missing_database_does_not_create_it(tmp_path, capsys):
    from types import SimpleNamespace

    from config import load_config

    db_path = tmp_path / "missing" / "dream.db"
    config = load_config(
        tmp_path / "config.toml",
        overrides={
            "storage": {"db_path": str(db_path), "memory_root": str(tmp_path / "memory")},
            "review": {"mode": "auto-apply"},
        },
    )
    args = SimpleNamespace(
        db=db_path,
        memory=str(tmp_path / "memory"),
        suggestions_cmd="curate-configured",
        config=config,
        dry_run=False,
    )

    assert dream.cmd_suggestions(args) == 0
    assert "No pending suggestions." in capsys.readouterr().out
    assert not db_path.exists()
