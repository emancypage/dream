"""Path identity and current retention guards for character operations."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from retention_policy import load_retention_policy, POLICY_FILE


def policy_for(config):
    return load_retention_policy(config.memory_root)


def check_constitution(document, policy):
    if policy.excludes(text=document.text):
        raise ValueError('constitution conflicts with retention policy')


def _same(a, b):
    if a.resolve() == b.resolve():
        return True
    try:
        return a.samefile(b)
    except (FileNotFoundError, OSError):
        return False


def is_constitution_source(source_path, *, memory_root=None, constitution_path):
    if constitution_path is None:
        return False
    source = Path(source_path).expanduser()
    if not source.is_absolute() and memory_root is not None:
        source = Path(memory_root) / source
    return _same(source, Path(constitution_path).expanduser())


def is_memory_control_source(source_path, *, memory_root, constitution_path):
    source = Path(source_path).expanduser()
    if not source.is_absolute():
        source = Path(memory_root) / source
    return (is_constitution_source(source, constitution_path=constitution_path)
            or _same(source, Path(memory_root) / POLICY_FILE))


def _check_artifact_inputs(config, manifest):
    from character.constitution import load_constitution
    from character.review import accepted_pairs
    from storage import open_db_readonly
    document = load_constitution(config)
    policy = policy_for(config)
    check_constitution(document, policy)
    if (manifest.get('constitution_sha256') != document.sha256
            or manifest.get('policy_fingerprint') != policy_fingerprint(policy)):
        raise ValueError('artifact inputs changed; render again')
    if manifest.get('pair_ids'):
        conn = open_db_readonly(config.db_path)
        try:
            current = {p['id']:p for p in accepted_pairs(conn, config, document)}
            for identity in manifest['pair_ids']:
                pair = current.get(identity)
                if (pair is None or pair['revision'] != manifest['pair_revisions'].get(identity)
                        or pair['content_sha256'] != manifest['pair_hashes'].get(identity)):
                    raise ValueError('accepted pair changed; render again')
        finally:
            conn.close()


def protected_paths(config, candidate=None):
    root = config.memory_root.expanduser()
    result = [Path(config.data['character']['constitution_path']).expanduser(), root / POLICY_FILE]
    bases = [root, Path.cwd(), config.db_path.expanduser().parent, Path(config.data['character']['state_root']).expanduser(), Path(config.data['character']['state_root']).expanduser() / 'backups']
    if candidate is not None:
        bases.append(Path(candidate).expanduser().parent)
    for base in bases:
        for location in (base, base.resolve()):
            for ancestor in (location, *location.parents):
                result.extend((ancestor / 'AGENTS.md', ancestor / 'CLAUDE.md'))
    for tree in (root, Path.home() / '.codex' / 'memories'):
        if tree.exists():
            result.extend(p for p in tree.rglob('*') if p.is_file())
    return result


def guard_path(path, config, *, output_root=None):
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError('symlink output is forbidden')
    resolved = candidate.resolve()
    memory = config.memory_root.expanduser().resolve()
    native = (Path.home() / '.codex' / 'memories').resolve()
    if resolved == memory or memory in resolved.parents or resolved == native or native in resolved.parents:
        raise ValueError('protected memory destination')
    if candidate.name in {'AGENTS.md', 'CLAUDE.md', POLICY_FILE}:
        raise ValueError('protected instruction destination')
    if any(_same(candidate, protected) for protected in protected_paths(config, candidate)):
        raise ValueError('protected destination identity')
    if output_root is not None:
        root = Path(output_root).expanduser().resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError('output escapes state root')
        if _same(candidate, config.db_path):
            raise ValueError('output aliases database')
    return resolved


def policy_fingerprint(policy):
    return hashlib.sha256(json.dumps(policy.__dict__, sort_keys=True).encode()).hexdigest()


def write_artifacts(config, files, manifest):
    _check_artifact_inputs(config, manifest)
    root = Path(config.data['character']['state_root']).expanduser()
    guard_path(root, config)
    targets = {}
    for name, body in files.items():
        target = Path(name)
        if not target.is_absolute():
            target = root / target
        targets[guard_path(target, config, output_root=root)] = body.encode('utf-8') if isinstance(body, str) else body
    if len(targets) != len(files):
        raise ValueError('duplicate output identities')
    paths = list(targets)
    if any(_same(a, b) for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('aliased output identities')
    initial = policy_fingerprint(policy_for(config))
    for body in targets.values():
        if policy_for(config).excludes(text=body.decode('utf-8')):
            raise ValueError('artifact conflicts with retention policy')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = []
    try:
        for target, body in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, tmp = tempfile.mkstemp(prefix='.character-', dir=target.parent)
            temp = Path(tmp)
            guard_path(temp, config, output_root=root)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((target, temp))
        if initial != policy_fingerprint(policy_for(config)):
            raise ValueError('retention policy changed; retry')
        _check_artifact_inputs(config, manifest)
        for target, temp in staged:
            _check_artifact_inputs(config, manifest)
            guard_path(target, config, output_root=root)
            if target.exists():
                backup_dir = root / 'backups'
                guard_path(backup_dir, config, output_root=root)
                backup_dir.mkdir(mode=0o700, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix=target.name + '.', dir=backup_dir)
                os.close(fd)
                backup = Path(name)
                guard_path(backup, config, output_root=root)
                shutil.copyfile(target, backup)
                backup.chmod(0o600)
            _check_artifact_inputs(config, manifest)
            guard_path(target, config, output_root=root)
            guard_path(temp, config, output_root=root)
            os.replace(temp, target)
    finally:
        for _, temp in staged:
            temp.unlink(missing_ok=True)
    return manifest
