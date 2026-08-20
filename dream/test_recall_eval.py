import json
from pathlib import Path

import pytest

from recall_eval import calibrate_fixture_file, evaluate_fixture_file
from recall_types import AdapterSettings, RecallSettings


def _settings():
    return RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False))


def test_public_fixture_evaluates(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    report = evaluate_fixture_file(Path(__file__).parent / "fixtures/recall/public.json", conn, _settings())
    assert report.query_count == 3
    assert report.forbidden_count == 0
    assert report.recall_at_1 >= 0


def test_public_fixture_is_rejected_for_calibration(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    with pytest.raises(ValueError):
        calibrate_fixture_file(Path(__file__).parent / "fixtures/recall/public.json", conn, "lexical")


def test_private_fixture_calibration_has_stable_version(tmp_path):
    fixture = tmp_path / "heldout.json"
    fixture.write_text(json.dumps({"documents": [{"id":"a","source_kind":"approved_memory","trust_level":"user_approved","project_slug":None,"source_path":"a.md","source_updated_at":"2026-01-01T00:00:00Z","source_version":"v1","text":"alpha"}], "queries": [{"id":"q","event":"prompt","query":"alpha","relevant":["a"]}]}), encoding="utf-8")
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    record = calibrate_fixture_file(fixture, conn, "lexical")
    assert record.mode == "lexical"
    assert len(record.calibration_version) == 64


def test_recall_eval_cli_uses_ephemeral_database(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_eval

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    db_dir.chmod(0o555)
    try:
        rc = cmd_recall_eval(argparse.Namespace(
            fixtures=Path(__file__).parent / "fixtures/recall/public.json",
            db=db_path,
            config=load_config(tmp_path / "missing.toml"),
        ))
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["query_count"] == 3
    finally:
        db_dir.chmod(0o755)
