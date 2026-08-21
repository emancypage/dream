from pathlib import Path


def test_recall_preflight_is_read_only(tmp_path):
    from config import load_config
    from recall_context import recall_preflight
    cfg = load_config(tmp_path / "config.toml")
    before = set(tmp_path.iterdir())
    checks = recall_preflight(cfg, tmp_path / "missing.db")
    assert checks
    assert set(tmp_path.iterdir()) == before


def test_recall_preflight_reads_schema_without_parent_write_access(tmp_path):
    from config import load_config
    from dream import open_db
    from recall_context import recall_preflight

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = open_db(db_path)
    conn.close()
    db_dir.chmod(0o555)
    try:
        checks = recall_preflight(load_config(tmp_path / "config.toml"), db_path)
        assert ("recall.index", True, "recall schema present") in checks
    finally:
        db_dir.chmod(0o755)


def test_empty_codex_memories_scaffold_is_not_duplicate_injection(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    (codex_home / "memories" / ".agents").mkdir(parents=True)
    (codex_home / "memories" / ".codex").mkdir()

    ok, detail = codex_memories_check(codex_home)

    assert ok
    assert "disabled" in detail or "empty" in detail


def test_enabled_codex_memories_is_reported_as_duplicate_injection(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "[features]\nmemories = true\n[memories]\nuse_memories = true\n",
        encoding="utf-8",
    )
    (codex_home / "memories").mkdir()

    ok, detail = codex_memories_check(codex_home)

    assert not ok
    assert "enabled" in detail


def test_explicitly_disabled_codex_memories_passes_even_with_old_files(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    memories = codex_home / "memories"
    memories.mkdir(parents=True)
    (memories / "old.md").write_text("preserved generated state", encoding="utf-8")
    (codex_home / "config.toml").write_text(
        "[features]\nmemories = false\n[memories]\n"
        "use_memories = false\ngenerate_memories = false\n",
        encoding="utf-8",
    )

    ok, detail = codex_memories_check(codex_home)

    assert ok
    assert "disabled" in detail


def test_mixed_codex_memories_controls_are_reported_as_enabled(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "[features]\nmemories = true\n[memories]\n"
        "use_memories = false\ngenerate_memories = true\n",
        encoding="utf-8",
    )

    ok, detail = codex_memories_check(codex_home)

    assert not ok
    assert "enabled" in detail


def test_structurally_invalid_codex_memories_config_returns_read_error(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text("features = true\n", encoding="utf-8")

    ok, detail = codex_memories_check(codex_home)

    assert not ok
    assert detail.startswith("Codex Memories configuration could not be read: ")
