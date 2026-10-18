import os
import dream


def test_memory_suggestion_cannot_overwrite_constitution_hardlink(tmp_path):
    root = tmp_path / 'memory'; root.mkdir()
    constitution = tmp_path / 'constitution.md'
    constitution.write_text('human-owned rules')
    os.link(constitution, root / 'alias.md')
    conn = dream.open_db(tmp_path / 'dream.db')
    row = conn.execute("INSERT INTO suggestions(kind,target_path,body) VALUES('update','alias.md','forged')")
    conn.commit()
    assert dream._apply_suggestion(conn, root, root / '.suggestions', row.lastrowid, 'update', 'alias.md', 'forged', None, constitution_path=constitution) is False
    assert constitution.read_text() == 'human-owned rules'
    conn.close()


def test_retention_index_cannot_alias_constitution(tmp_path):
    import json
    import pytest
    from config import load_config
    from retention_cleanup import apply_retention
    root = tmp_path / 'memory'; root.mkdir()
    constitution = tmp_path / 'rules.md'
    text = 'Human rules\n- [Secret](secret.md)\n'
    constitution.write_text(text)
    os.link(constitution, root / 'MEMORY.md')
    (root / 'secret.md').write_text('retired detail')
    (root / '.dream-retention.json').write_text(json.dumps({'memory_paths': ['secret.md']}))
    cfg = load_config(tmp_path / 'absent', {'storage': {'db_path': str(tmp_path / 'dream.db'), 'memory_root': str(root)}, 'character': {'constitution_path': str(constitution), 'state_root': str(tmp_path / 'state')}})
    conn = dream.open_db(cfg.db_path)
    with pytest.raises(ValueError, match='protected'):
        apply_retention(conn, root, config=cfg, apply=True)
    assert constitution.read_text() == text
    conn.close()


def test_index_prune_cannot_alias_constitution_or_policy(tmp_path):
    import json
    import pytest
    root = tmp_path / 'memory'; root.mkdir()
    constitution = tmp_path / 'rules.md'
    constitution.write_text('Human rules\n- [Gone](gone.md)\n')
    os.link(constitution, root / 'MEMORY.md')
    with pytest.raises(ValueError, match='protected'):
        dream._prune_orphaned_index_lines(root, constitution_path=constitution)
    assert constitution.read_text().startswith('Human rules')
    (root / 'MEMORY.md').unlink()
    policy = root / '.dream-retention.json'
    original = json.dumps({'instructions': '[Gone](gone.md)'})
    policy.write_text(original)
    os.link(policy, root / 'MEMORY.md')
    with pytest.raises(ValueError, match='protected'):
        dream._prune_orphaned_index_lines(root, constitution_path=constitution)
    assert policy.read_text() == original
