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
