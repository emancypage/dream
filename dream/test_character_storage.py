import sqlite3
from pathlib import Path
from character.storage import insert_record, records

def test_payload_roundtrip():
    conn = sqlite3.connect(':memory:')
    conn.executescript((Path(__file__).parent / 'schema.sql').read_text())
    payload = {'id': 'p1', 'revision': 1, 'content_sha256': 'abc', 'chosen': 'yes', 'nested': {'exact': [1, 'a']}}
    insert_record(conn, 'character_pairs', payload)
    stored = records(conn, 'character_pairs')[0]
    assert {k: v for k, v in stored.items() if k != 'content_sha256'} == {k: v for k, v in payload.items() if k != 'content_sha256'}
    assert stored['content_sha256'] != 'abc'

def test_migration_backup_precedes_character_ddl(tmp_path):
    from dream import open_db
    path = tmp_path / 'dream.db'
    conn = open_db(path)
    conn.execute('DELETE FROM schema_migrations WHERE version=3')
    for name in ('constitutions', 'moments', 'seeds', 'pairs', 'judgments', 'review_events', 'eval_runs', 'eval_results'):
        conn.execute(f'DROP TABLE character_{name}')
    conn.commit()
    conn.close()
    conn = open_db(path)
    assert conn.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0] == 3
    backups = list(tmp_path.glob('dream.db.bak-*'))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    assert backup.execute("SELECT name FROM sqlite_master WHERE name='character_pairs'").fetchone() is None
    assert backup.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0] == 2
    backup.close()
    conn.close()


def test_nested_policy_and_pair_manifest_lineage(tmp_path):
    import json
    from config import load_config
    from character.storage import eligible_records
    memory = tmp_path / 'memory'
    memory.mkdir()
    (memory / '.dream-retention.json').write_text(json.dumps({'text_markers': ['private-marker']}))
    config = load_config(tmp_path / 'absent', {'storage': {'memory_root': str(memory)}})
    conn = sqlite3.connect(':memory:')
    conn.executescript((Path(__file__).parent / 'schema.sql').read_text())
    insert_record(conn, 'character_pairs', {'id': 'p1', 'nested': {'rationale': 'private-marker'}})
    insert_record(conn, 'character_eval_runs', {'id': 'run1', 'render': {'pair_ids': ['p1']}})
    insert_record(conn, 'character_eval_results', {'id': 'result1', 'run_id': 'run1'})
    assert eligible_records(conn, config, 'character_eval_results') == []


def test_updates_recompute_hash_and_require_new_revision():
    import pytest
    from character.storage import update_record
    conn = sqlite3.connect(':memory:')
    conn.executescript((Path(__file__).parent / 'schema.sql').read_text())
    old = insert_record(conn, 'character_pairs', {'id': 'p', 'chosen': 'first'})
    changed = dict(old, chosen='second', revision=2)
    updated = update_record(conn, 'character_pairs', changed, expected_sha=old['content_sha256'])
    assert updated['content_sha256'] != old['content_sha256']
    with pytest.raises(ValueError):
        update_record(conn, 'character_pairs', dict(updated, revision=3), expected_sha=old['content_sha256'])
    with pytest.raises(ValueError):
        update_record(conn, 'character_pairs', updated)
