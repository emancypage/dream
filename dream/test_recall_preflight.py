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


def test_invalid_utf8_codex_memories_config_returns_read_error(tmp_path):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_bytes(b"[features]\nmemories = true\n\xff\n")

    ok, detail = codex_memories_check(codex_home)

    assert not ok
    assert detail.startswith("Codex Memories configuration could not be read: ")


def test_codex_memories_config_existence_oserror_returns_read_error(tmp_path, monkeypatch):
    from recall_context import codex_memories_check

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    original_exists = Path.exists

    def raising_exists(path):
        if path == config_path:
            raise OSError("config stat failed")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", raising_exists)

    ok, detail = codex_memories_check(codex_home)

    assert not ok
    assert detail.startswith("Codex Memories configuration could not be read: ")


def _embedder_check(tmp_path, body):
    from config import load_config
    from recall_context import recall_preflight
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    checks = recall_preflight(load_config(path), tmp_path / "missing.db")
    return next((ok, detail) for name, ok, detail in checks if name == "recall.embedder")


def _embedder_table(url, **extra):
    from test_recall_embedder_http import DIMENSION, MODEL
    values = {"enabled": "true", "type": '"openai-compatible"', "base_url": f'"{url}"', "model": f'"{MODEL}"', "dimension": DIMENSION}
    values.update(extra)
    return "[recall.embedder]\n" + "".join(f"{key} = {value}\n" for key, value in values.items())


def test_preflight_reports_a_disabled_embedder_without_network(tmp_path):
    assert _embedder_check(tmp_path, "") == (True, "disabled")


def test_preflight_reports_a_reachable_embedder_with_server_model_and_dimension(tmp_path):
    from test_recall_embedder_http import DIMENSION, MODEL, FakeEmbeddingServer
    server = FakeEmbeddingServer().start()
    try:
        ok, detail = _embedder_check(tmp_path, _embedder_table(server.url))
        assert ok
        assert detail.startswith("reachable") and MODEL in detail and f"dimension {DIMENSION}" in detail
        # One probe string, never a document.
        assert server.posted_inputs() == ["dream preflight probe"]
    finally:
        server.stop()


def test_preflight_reports_an_unreachable_embedder(tmp_path):
    from test_recall_embedder_http import refused_url
    ok, detail = _embedder_check(tmp_path, _embedder_table(refused_url(), timeout_seconds=0.5))
    assert not ok and detail.startswith("unreachable")


def test_preflight_reports_a_server_model_mismatch_as_misconfigured(tmp_path):
    from test_recall_embedder_http import FakeEmbeddingServer
    server = FakeEmbeddingServer(model="bge-m3-q8").start()
    try:
        ok, detail = _embedder_check(tmp_path, _embedder_table(server.url))
        assert not ok and detail.startswith("misconfigured") and "bge-m3-q8" in detail
    finally:
        server.stop()


def test_preflight_reports_invalid_embedder_config_as_misconfigured(tmp_path):
    ok, detail = _embedder_check(tmp_path, _embedder_table("http://127.0.0.1:8081", dimension=0))
    assert not ok and detail.startswith("misconfigured") and "dimension" in detail
    ok, detail = _embedder_check(tmp_path, _embedder_table("http://192.0.2.10:8081"))
    assert not ok and detail.startswith("misconfigured") and "remote_data_egress" in detail
    ok, detail = _embedder_check(tmp_path, '[recall.embedder]\nenabled = false\nbase_url = "http://evil.com@127.0.0.1:8081"\n')
    assert not ok and detail.startswith("misconfigured")
