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
