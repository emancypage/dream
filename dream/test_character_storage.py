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
    conn.execute('DELETE FROM schema_migrations WHERE version>=3')
    for name in ('constitutions', 'moments', 'seeds', 'pairs', 'judgments', 'review_events', 'eval_runs', 'eval_results', 'short_examples'):
        conn.execute(f'DROP TABLE character_{name}')
    conn.commit()
    conn.close()
    conn = open_db(path)
    assert conn.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0] == 4
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


def test_successful_write_observes_constitution_metadata(tmp_path):
    from types import SimpleNamespace
    import dream
    from character import storage
    conn = dream.open_db(tmp_path / 'dream.db')
    doc = SimpleNamespace(sha256='constitution-hash', version='human-v1', approved_at='2026-10-18T10:00:00Z')
    storage.observe_constitution(conn, doc)
    storage.observe_constitution(conn, doc)
    observations = storage.records(conn, 'character_constitutions')
    assert len(observations) == 1
    assert observations[0]['constitution_version'] == 'human-v1'
    assert observations[0]['approved_at'] == doc.approved_at
    assert 'text' not in observations[0]


def _writer_config(tmp_path):
    from config import load_config
    return load_config(tmp_path / 'absent', {
        'storage': {'db_path': str(tmp_path / 'dream.db'), 'memory_root': str(tmp_path / 'memory')},
        'character': {'state_root': str(tmp_path / 'state'), 'constitution_path': str(tmp_path / 'rules')},
    })


def test_ordinary_writes_never_create_backups(tmp_path):
    from character.storage import open_writer
    config = _writer_config(tmp_path)
    for index in range(4):
        with open_writer(config) as writer:
            insert_record(writer, 'character_pairs', {'id': f'p{index}', 'chosen': 'yes'})
    assert not (tmp_path / 'state' / 'backups').exists()


def test_nightly_backup_is_small_bounded_and_once_per_day_across_processes(tmp_path):
    import json
    import os
    import stat
    import subprocess
    import sys
    from datetime import date
    from dream import open_db

    config = _writer_config(tmp_path)
    conn = open_db(config.db_path)
    insert_record(conn, 'character_pairs', {'id': 'p1', 'chosen': 'yes'})
    conn.execute('CREATE TABLE unrelated_large_data (body BLOB)')
    conn.execute('INSERT INTO unrelated_large_data VALUES (zeroblob(2000000))')
    conn.commit()
    conn.close()
    config_path = tmp_path / 'config.toml'
    config_path.write_text(
        f'[storage]\ndb_path = "{config.db_path}"\nmemory_root = "{config.memory_root}"\n'
        f'[character]\nstate_root = "{tmp_path / "state"}"\nbackup_keep = 2\n')
    backups = tmp_path / 'state' / 'backups'
    backups.mkdir(parents=True)
    for day in ('2000-01-01', '2000-01-02', '2000-01-03'):
        (backups / f'character-{day}.json').write_text('{}')
    env = {**os.environ, 'DREAM_CONTROL_STATE_DIR': str(tmp_path / 'control')}
    command = [sys.executable, str(Path(__file__).with_name('dream.py')),
               '--config', str(config_path), 'character', 'backup']
    first = subprocess.run(command, capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    backup = backups / f'character-{date.today().isoformat()}.json'
    assert sorted(p.name for p in backups.iterdir()) == ['character-2000-01-03.json', backup.name]
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    content = backup.read_bytes()
    assert len(content) < 64000
    data = json.loads(content)
    assert [row['id'] for row in data['character_pairs']] == ['p1']
    assert set(data) == {'character_constitutions', 'character_moments', 'character_seeds',
                         'character_pairs', 'character_judgments', 'character_review_events',
                         'character_eval_runs', 'character_eval_results', 'character_short_examples'}
    conn = open_db(config.db_path)
    insert_record(conn, 'character_pairs', {'id': 'p2', 'chosen': 'later'})
    conn.commit()
    conn.close()
    before = backup.stat()
    second = subprocess.run(command, capture_output=True, text=True, env=env)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout)['skipped'] == 'already backed up today'
    assert len(list(backups.iterdir())) == 2
    assert backup.read_bytes() == content
    assert backup.stat().st_mtime_ns == before.st_mtime_ns


def test_nightly_backup_skips_missing_database_without_creating_it(tmp_path):
    from character.storage import backup_daily
    config = _writer_config(tmp_path)
    assert backup_daily(config) == {'skipped': 'database missing'}
    assert not config.db_path.exists()
    assert not (tmp_path / 'state').exists()
