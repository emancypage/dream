import pytest
from config import load_config, ConfigError

def test_character_defaults(tmp_path):
    cfg = load_config(tmp_path / 'absent')
    assert cfg.data['character']['enabled'] is False
    assert cfg.data['character']['render_max_pairs'] == 3
    assert cfg.stage('review')['model'] == 'gpt-5.6-luna'

@pytest.mark.parametrize('value', [{'auto_apply': True}, {'render_max_pairs': True}, {'enabled': 'yes'}])
def test_strict_character(tmp_path, value):
    with pytest.raises(ConfigError):
        load_config(tmp_path / 'absent', {'character': value})


def test_same_family_routes_require_explicit_override(tmp_path):
    with pytest.raises(ConfigError, match='family'):
        load_config(tmp_path / 'absent', {'character': {'enabled': True}, 'stages': {'character_judge': {'provider': 'codex'}}})
    cfg = load_config(tmp_path / 'absent', {'character': {'enabled': True, 'allow_same_family_judge': True}, 'stages': {'character_judge': {'provider': 'codex'}}})
    assert cfg.data['character']['allow_same_family_judge']

@pytest.mark.parametrize('stage', [
    {'model': 2}, {'timeout_seconds': True}, {'timeout_seconds': 0}, {'reasoning_effort': []},
])
def test_strict_character_stage_fields(tmp_path, stage):
    with pytest.raises(ConfigError):
        load_config(tmp_path / 'absent', {'stages': {'character_extract': stage}})


def test_preflight_ignores_disabled_character_and_scopes_explicit_stages(tmp_path, monkeypatch):
    import backend
    seen = []
    monkeypatch.setattr(backend, 'preflight_provider', lambda cfg: (seen.append(cfg['type']) or True, 'ok'))
    cfg = load_config(tmp_path / 'absent')
    backend.preflight(cfg)
    assert seen == ['codex-cli']
    seen.clear()
    backend.preflight(cfg, stages=['character_judge'])
    assert seen == ['claude-cli']


def test_unknown_character_provider_auth_rejected_even_with_override(tmp_path):
    with pytest.raises(ConfigError, match='family|auth'):
        load_config(tmp_path / 'absent', {
            'character': {'enabled': True, 'allow_same_family_judge': True},
            'providers': {'codex': {'auth': 'unknown-auth'}},
        })


def test_explicit_unknown_claude_auth_rejected(tmp_path):
    with pytest.raises(ConfigError, match='family|auth'):
        load_config(tmp_path / 'absent', {
            'character': {'enabled': True},
            'providers': {'claude': {'auth': 'unknown-auth'}},
        })


def test_constitution_path_cannot_be_inside_generated_memory(tmp_path):
    root = tmp_path / 'memory'
    with pytest.raises(ConfigError, match='outside'):
        load_config(tmp_path / 'absent', {'storage': {'memory_root': str(root)}, 'character': {'constitution_path': str(root / 'rules.md')}})
