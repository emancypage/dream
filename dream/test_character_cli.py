import json
import dream
from config import load_config


def test_disabled_nightly_character_commands_do_not_open_database(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[storage]\ndb_path="' + str(tmp_path / 'never.db') + '"\n')
    monkeypatch.setattr(dream, 'open_db', lambda *a: (_ for _ in ()).throw(AssertionError('writer opened')))
    for argv in (['generate', '--source', 'all', '--limit', '10'], ['judge', '--limit', '10']):
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
