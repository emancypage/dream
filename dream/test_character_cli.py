import json
import dream
from config import load_config


def test_disabled_nightly_character_commands_do_not_open_database(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[storage]\ndb_path="' + str(tmp_path / 'never.db') + '"\n[character]\nconstitution_path="' + str(tmp_path / 'absent-rules') + '"\n')
    monkeypatch.setattr(dream, 'open_db', lambda *a: (_ for _ in ()).throw(AssertionError('writer opened')))
    for argv in (['generate', '--source', 'all', '--limit', '10'], ['judge', '--limit', '10'], ['eval', '--fixtures', str(tmp_path / 'absent-probes'), '--limit', '1']):
        assert dream.main(['--config', str(cfg), 'character', *argv]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result['skipped'] == 'character disabled'
    assert not (tmp_path / 'never.db').exists()


def test_character_help_lists_manual_and_deployment_commands(capsys):
    import pytest
    with pytest.raises(SystemExit) as exc:
        dream.main(['character', '--help'])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for name in ('estimate', 'generate', 'judge', 'accept', 'edit', 'render', 'eval', 'export'):
        assert name in output


def test_eval_report_reopens_snapshot_after_writing_results(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace
    from character import cli, storage
    db_path = tmp_path / 'dream.db'
    dream.open_db(db_path).close()
    cfg_path = tmp_path / 'config.toml'
    cfg_path.write_text('[storage]\ndb_path="' + str(db_path) + '"\n[character]\nenabled=true\n')
    def evaluate(conn, config, **kwargs):
        assert conn.execute('SELECT count(*) FROM character_eval_runs').fetchone()[0] == 0
        writer = dream.open_db(db_path)
        storage.insert_record(writer, 'character_eval_runs', {'id': 'run-one'})
        writer.commit(); writer.close()
        return {'run_id': 'run-one', 'failures': []}
    def report(conn, config, run_id):
        assert conn.execute('SELECT count(*) FROM character_eval_runs').fetchone()[0] == 1
        return {'markdown': 'a comparison', 'manifest': {}}
    monkeypatch.setitem(sys.modules, 'character.eval', SimpleNamespace(evaluate=evaluate, report_run=report))
    monkeypatch.setattr(cli, '_save', lambda *a: None)
    assert dream.main(['--config', str(cfg_path), 'character', 'eval', '--fixtures', str(tmp_path / 'fixture.json'), '--limit', '1']) == 0


def test_enabled_writer_migrates_v2_but_readers_do_not(tmp_path, monkeypatch, capsys):
    import hashlib
    import sqlite3
    from character import storage
    from character import generate
    from test_character_constitution import TEXT
    db = tmp_path / 'dream.db'
    conn = dream.open_db(db)
    for table in storage.TABLES: conn.execute('DROP TABLE ' + table)
    conn.execute('DELETE FROM schema_migrations WHERE version>=3'); conn.commit(); conn.close()
    constitution = tmp_path / 'rules.md'; constitution.write_text(TEXT)
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[storage]\ndb_path="' + str(db) + '"\n[character]\nenabled=true\nconstitution_path="' + str(constitution) + '"\nstate_root="' + str(tmp_path / 'state') + '"\n')
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert dream.main(['--config', str(cfg), 'character', 'list']) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob('dream.db.bak-*'))
    def fake(conn, config, **kwargs):
        assert conn.execute('SELECT count(*) FROM character_pairs').fetchone()[0] == 0
        return {'generated': 0, 'failures': []}
    monkeypatch.setattr(generate, 'generate_pairs', fake)
    assert dream.main(['--config', str(cfg), 'character', 'generate', '--limit', '1']) == 0
    assert list(tmp_path.glob('dream.db.bak-*'))
    with sqlite3.connect(db) as check:
        assert check.execute('SELECT max(version) FROM schema_migrations').fetchone()[0] == 4


import pytest

@pytest.mark.parametrize('command', [['render'], ['export', '--format', 'dpo-jsonl']])
def test_saved_render_and_export_never_migrate_database(tmp_path, command):
    import hashlib
    from character import storage
    from test_character_constitution import TEXT
    db = tmp_path / 'dream.db'
    conn = dream.open_db(db)
    for table in storage.TABLES: conn.execute('DROP TABLE ' + table)
    conn.execute('DELETE FROM schema_migrations WHERE version>=3'); conn.commit(); conn.close()
    constitution = tmp_path / 'rules.md'; constitution.write_text(TEXT)
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[storage]\ndb_path="' + str(db) + '"\n[character]\nenabled=true\nconstitution_path="' + str(constitution) + '"\nstate_root="' + str(tmp_path / 'state') + '"\n')
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert dream.main(['--config', str(cfg), 'character', *command]) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob('dream.db.bak-*'))
    assert not (tmp_path / 'state').exists()
