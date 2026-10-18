import os
import pytest
from config import load_config
from character.safety import guard_path

def test_protected_aliases(tmp_path):
    constitution = tmp_path / 'constitution.md'
    constitution.write_text('human')
    cfg = load_config(tmp_path / 'absent', {'character': {'constitution_path': str(constitution)}})
    alias = tmp_path / 'alias'
    os.link(constitution, alias)
    with pytest.raises(ValueError):
        guard_path(alias, cfg)
    alias.unlink()
    alias.symlink_to(constitution)
    with pytest.raises(ValueError):
        guard_path(alias, cfg)

@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'dotdot'])
def test_guard_protected_db_and_output(tmp_path, kind):
    constitution = tmp_path / 'constitution.md'
    constitution.write_text('human')
    alias = tmp_path / 'alias.db'
    if kind == 'symlink':
        alias.symlink_to(constitution)
    elif kind == 'hardlink':
        os.link(constitution, alias)
    else:
        (tmp_path / 'child').mkdir()
        alias = tmp_path / 'child' / '..' / 'constitution.md'
    cfg = load_config(tmp_path / 'absent', {'storage': {'db_path': str(alias)}, 'character': {'constitution_path': str(constitution)}})
    from character.storage import open_writer
    with pytest.raises(ValueError):
        with open_writer(cfg):
            pass
    with pytest.raises(ValueError):
        guard_path(alias, cfg, output_root=tmp_path)
    assert constitution.read_text() == 'human'


def test_state_root_escape(tmp_path):
    cfg = load_config(tmp_path / 'absent')
    with pytest.raises(ValueError, match='escapes'):
        guard_path(tmp_path / 'outside', cfg, output_root=tmp_path / 'state')


def test_database_hardlink_to_adjacent_instruction(tmp_path):
    instruction = tmp_path / 'AGENTS.md'
    instruction.write_text('human instructions')
    db = tmp_path / 'character.db'
    os.link(instruction, db)
    cfg = load_config(tmp_path / 'absent', {'storage': {'db_path': str(db)}})
    with pytest.raises(ValueError, match='protected'):
        guard_path(db, cfg)
    assert instruction.read_text() == 'human instructions'
