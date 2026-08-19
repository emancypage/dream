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
