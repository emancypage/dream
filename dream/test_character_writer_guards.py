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
