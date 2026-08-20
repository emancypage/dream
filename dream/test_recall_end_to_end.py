from pathlib import Path

from recall_context import run_context
from recall_documents import synchronize_recall_documents
from recall_types import AdapterSettings, RecallQuery, RecallSettings


def test_memory_file_reaches_context_with_provenance(tmp_path):
    from dream import open_db
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "api.md").write_text("Postgres migration decision.", encoding="utf-8")
    conn = open_db(tmp_path / "dream.db")
    report = synchronize_recall_documents(conn, memory, include_raw_transcripts=False)
    assert report.inserted == 1
    settings = RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False))
    query = RecallQuery("Postgres migration", "s1", "session-start:startup", str(memory), (str(memory),), 6000, frozenset(), False)
    result = run_context(conn, query, settings)
    assert "Dream recall" in result.rendered_context
    assert "approved_memory" in result.rendered_context


def test_prompt_hook_reads_database_without_writing_to_it(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from config import load_config
    from dream import cmd_context, open_db

    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "api.md").write_text("Postgres migration decision.", encoding="utf-8")
    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = open_db(db_path)
    synchronize_recall_documents(conn, memory, include_raw_transcripts=False)
    conn.close()

    state = tmp_path / "hook-state"
    monkeypatch.setenv("DREAM_RECALL_STATE_ROOT", str(state))
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(json.dumps({
            "session_id": "sandbox-session",
            "prompt": "Postgres migration",
            "cwd": str(memory),
            "repository_roots": [str(memory)],
        })),
    )
    db_dir.chmod(0o555)
    try:
        rc = cmd_context(SimpleNamespace(
            context_cmd="prompt",
            explain=False,
            config=load_config(tmp_path / "config.toml"),
            db=db_path,
        ))
        output = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert output["continue"] is True
        assert "Dream recall" in output["hookSpecificOutput"]["additionalContext"]
        assert not list(db_dir.glob("dream.db-*"))
        assert list(state.glob("*.json"))
    finally:
        db_dir.chmod(0o755)
