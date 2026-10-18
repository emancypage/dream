"""Delete excluded derived character data after a complete backup snapshot."""
import base64
import json
from pathlib import Path

from character import storage
from character.safety import guard_path, policy_for, policy_fingerprint
from character.types import canonical_json, content_hash


def collect(conn, config):
    """Return affected records and managed file bytes without changing anything."""
    all_rows = storage.snapshot(conn)
    affected = {}
    for table, rows in all_rows.items():
        allowed = {row['id'] for row in storage.eligible_records(conn, config, table)}
        affected[table] = [row for row in rows if row['id'] not in allowed]
    ids = {row['id'] for rows in affected.values() for row in rows}
    root = Path(config.data['character']['state_root']).expanduser()
    guard_path(root, config)
    files = {}
    expected = {}
    policy = policy_for(config)
    for path in sorted(root.glob('*.manifest.json')):
        guard_path(path, config, output_root=root)
        raw = path.read_bytes()
        try:
            manifest = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        def references(value):
            if isinstance(value, str):
                return value in ids
            if isinstance(value, dict):
                return any(references(v) for v in value.values())
            if isinstance(value, list):
                return any(references(v) for v in value)
            return False
        if not references(manifest) and not policy.excludes(text=raw.decode('utf-8')):
            continue
        files[path.name] = base64.b64encode(raw).decode('ascii')
        expected[path.name] = content_hash({'bytes': files[path.name]})
        for name, expected_sha in manifest.get('files', {}).items():
            if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
                raise ValueError('invalid managed artifact name')
            target = guard_path(root / name, config, output_root=root)
            if target.is_file():
                files[name] = base64.b64encode(target.read_bytes()).decode('ascii')
                expected[name] = expected_sha
    return {'rows': affected, 'files': files, 'expected_files': expected,
            'policy_fingerprint': policy_fingerprint(policy)}


def preview(snapshot):
    return {'rows': {table: len(rows) for table, rows in snapshot['rows'].items()},
            'managed_files': sorted(snapshot['files']), 'rerun_required': False,
            'modified_files': []}


def apply_snapshot(conn, config, original):
    """All character rows delete together, or none do when the snapshot changed."""
    result = preview(original)
    if conn.in_transaction:
        raise ValueError('character retention requires a clean transaction boundary')
    conn.execute('BEGIN IMMEDIATE')
    try:
        current = collect(conn, config)
        if content_hash(current) != content_hash(original):
            conn.rollback()
            result['rerun_required'] = True
            return result
        for table in reversed(storage.TABLES):
            for row in original['rows'][table]:
                conn.execute(f'DELETE FROM {table} WHERE id=?', (row['id'],))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    root = Path(config.data['character']['state_root']).expanduser()
    # Preserve files changed by the human, including before the backup snapshot.
    for name, encoded in original['files'].items():
        path = guard_path(root / name, config, output_root=root)
        saved = base64.b64decode(encoded)
        if not path.exists():
            continue
        import hashlib
        expected = original['expected_files'][name]
        is_manifest = name.endswith('.manifest.json')
        expected_matches = content_hash({'bytes': encoded}) == expected if is_manifest else hashlib.sha256(saved).hexdigest() == expected
        if path.read_bytes() != saved or not expected_matches:
            result['modified_files'].append(name)
            continue
        guard_path(path, config, output_root=root)
        path.unlink()
    return result
