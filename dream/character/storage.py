"""Canonical per-table payload storage with host identities."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import tempfile
import os
from character.types import canonical_json, content_hash
from character.safety import guard_path, policy_for

TABLES = ('character_constitutions', 'character_moments', 'character_seeds', 'character_pairs', 'character_judgments', 'character_review_events', 'character_eval_runs', 'character_eval_results')

def _table(table):
    if table not in TABLES:
        raise ValueError('invalid character table')
    return table

def records(conn, table):
    import json
    try:
        rows = conn.execute(f'SELECT payload_json FROM {_table(table)} ORDER BY id').fetchall()
    except sqlite3.OperationalError as exc:
        if 'no such table' in str(exc):
            raise ValueError('character schema missing; initialize through a writing command') from exc
        raise
    return [json.loads(row[0]) for row in rows]

def get_record(conn, table, id):
    return next((row for row in records(conn, table) if row['id'] == id), None)

def _values(record):
    if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id']:
        raise ValueError('record requires host ID')
    revision = record.get('revision', 1)
    if type(revision) is not int or revision < 1:
        raise ValueError('invalid record revision')
    payload = dict(record)
    payload.setdefault('revision', revision)
    payload['content_sha256'] = content_hash({key: value for key, value in payload.items() if key != 'content_sha256'})
    attempt = payload.get('generation_attempt', 1)
    work = payload.get('work_key')
    index_key = f'{work}:{attempt}' if work is not None and 'generation_attempt' in payload else work
    return payload['id'], index_key, revision, payload['content_sha256'], canonical_json(payload)

def insert_record(conn, table, record):
    conn.execute(f'INSERT INTO {_table(table)} (id,work_key,revision,content_sha256,payload_json) VALUES (?,?,?,?,?)', _values(record))
    return get_record(conn, table, record['id'])

def update_record(conn, table, record, *, expected_sha=None):
    old = get_record(conn, table, record['id'])
    if old is None or (expected_sha is not None and old['content_sha256'] != expected_sha):
        raise ValueError('record changed; retry')
    values = _values(record)
    if values[2] <= old['revision']:
        raise ValueError('record revision must increase')
    changed = conn.execute(f'UPDATE {_table(table)} SET work_key=?,revision=?,content_sha256=?,payload_json=? WHERE id=? AND content_sha256=? AND revision=?', (*values[1:], values[0], old['content_sha256'], old['revision']))
    if changed.rowcount != 1:
        raise ValueError('record changed; retry')
    return get_record(conn, table, record['id'])

def snapshot(conn):
    return {table: records(conn, table) for table in TABLES}

def _text_allowed(value, policy):
    if isinstance(value, str):
        return not policy.excludes(text=value)
    if isinstance(value, dict):
        if policy.excludes(project_slug=value.get('project_slug'), source_path=value.get('source_path', '')):
            return False
        return all(_text_allowed(v, policy) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_text_allowed(v, policy) for v in value)
    return True

def eligible_records(conn, config, table, constitution=None):
    policy = policy_for(config)
    data = snapshot(conn)
    def source_allowed(value):
        if isinstance(value, dict):
            path = value.get('source_path')
            if path and policy.excludes_memory_path(config.memory_root, path):
                return False
            session = value.get('session_id')
            if session:
                source = conn.execute('SELECT project_slug FROM sessions WHERE session_id=?', (session,)).fetchone()
                if source and policy.excludes(project_slug=source[0]):
                    return False
            return all(source_allowed(v) for v in value.values())
        if isinstance(value, list):
            return all(source_allowed(v) for v in value)
        return True
    blocked = {row['id'] for rows in data.values() for row in rows if row.get('excluded') or not _text_allowed(row, policy) or not source_allowed(row)}
    def dependencies(value, key=''):
        if isinstance(value, dict):
            return set().union(*(dependencies(v, k) for k,v in value.items())) if value else set()
        if isinstance(value, list):
            return set().union(*(dependencies(v, key) for v in value)) if value else set()
        return {value} if isinstance(value, str) and (key.endswith('_id') or key.endswith('_ids')) else set()
    while True:
        more = {row['id'] for rows in data.values() for row in rows if dependencies(row) & blocked}
        if more <= blocked:
            break
        blocked |= more
    return [row for row in data[_table(table)] if row['id'] not in blocked and (constitution is None or row.get('constitution_sha256') == constitution.sha256)]

@contextmanager
def open_writer(config):
    from dream import open_db
    path = guard_path(config.db_path, config)
    root = Path(config.data['character']['state_root']).expanduser()
    guard_path(root, config)
    conn = open_db(path)
    try:
        existing = any(records(conn, table) for table in TABLES)
        if existing:
            backup_root = root / 'backups'
            guard_path(backup_root, config)
            backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, name = tempfile.mkstemp(prefix='character-', suffix='.db', dir=backup_root)
            os.close(fd)
            backup_path = guard_path(name, config)
            dest = sqlite3.connect(backup_path)
            try:
                conn.backup(dest)
            finally:
                dest.close()
        with conn:
            yield conn
    finally:
        conn.close()
