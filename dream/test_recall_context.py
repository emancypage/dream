import io
import json
import sys

from recall_context import hook_success_json, parse_hook_payload, run_context
from recall_types import AdapterSettings, RecallSettings


def _settings():
    return RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False))


def test_packaged_session_budget_keeps_character_and_recalled_memory(tmp_path):
    from config import load_config
    from recall_context import compose_hook_context
    from recall_hooks import install_hooks

    settings = load_config(tmp_path / "config.toml").recall_settings()
    character = "# Trusted character\n" + "x" * 11980
    memory = "[Dream recall]\n" + "m" * 6000
    combined, reason = compose_hook_context(
        memory, character, settings.session_start_budget_codepoints,
        settings.session_start_additional_context_limit,
    )
    assert reason is None
    assert combined.startswith(character)
    assert "[Dream recall]" in combined
    assert len(combined.encode("utf-8")) <= 18000

    hooks_path = tmp_path / "hooks.json"
    install_hooks(hooks_path)
    hook = json.loads(hooks_path.read_text())["hooks"]["SessionStart"][0]["hooks"][0]
    assert len(combined.encode("utf-8")) <= hook["additionalContextLimit"]


def test_parse_hook_payload_and_exact_success_json():
    query = parse_hook_payload("UserPromptSubmit", {"session_id": "s", "prompt": "hello", "cwd": "/tmp", "repository_roots": ["/tmp"]})
    assert query.hook_event == "prompt"


def test_hook_success_json_includes_codex_event_name():
    assert json.loads(hook_success_json("UserPromptSubmit", "ctx")) == {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "ctx",
        },
    }
    assert json.loads(hook_success_json("SessionStart", "ctx")) == {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "ctx",
        },
    }


def test_missing_hook_fields_fail_open_parser():
    import pytest
    with pytest.raises(ValueError):
        parse_hook_payload("SessionStart", {"cwd": "/tmp"})


def test_context_explain_has_bounded_diagnostics(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    result = run_context(conn, parse_hook_payload("UserPromptSubmit", {"session_id": "s", "prompt": "missing", "cwd": "/tmp"}), _settings(), explain=True, constitution_path=None)
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


def test_constitution_guard_agrees_with_is_constitution_source(tmp_path, monkeypatch):
    """The per-run guard resolves the constitution once and must catch the same paths."""
    import os
    from character.safety import is_constitution_source
    from recall_context import constitution_guard

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "memory"
    (root / "sub").mkdir(parents=True)
    constitution = tmp_path / "rules" / "constitution.md"
    constitution.parent.mkdir()
    constitution.write_text("rules", encoding="utf-8")
    (root / "notes.md").write_text("notes", encoding="utf-8")
    (root / "link.md").symlink_to(constitution)
    os.link(constitution, root / "sub" / "hard.md")
    (tmp_path / "rules" / "alias").symlink_to(constitution.parent, target_is_directory=True)
    sources = ["notes.md", "link.md", "sub/hard.md", "missing.md", "sub/../link.md", str(constitution),
               "~/rules/constitution.md", str(tmp_path / "rules" / "alias" / "constitution.md"), "distilled:codex:x"]
    targets = [constitution, tmp_path / "rules" / "alias" / "constitution.md", "~/rules/constitution.md",
               tmp_path / "absent.md", None]
    for memory_root in (root, None):
        for target in targets:
            guard = constitution_guard(memory_root, target)
            for source in sources:
                expected = is_constitution_source(source, memory_root=memory_root, constitution_path=target)
                assert guard(source) == expected, (memory_root, target, source)
    assert constitution_guard(root, constitution)("link.md") and constitution_guard(root, constitution)("sub/hard.md")
