import hashlib
import json
from pathlib import Path
import pytest
import dream
from config import load_config
from character.storage import insert_record, records
from retention_cleanup import apply_retention


@pytest.fixture
def setup(tmp_path):
    memory = tmp_path / 'memory'; memory.mkdir()
    state = tmp_path / 'state'; state.mkdir()
    cfg = load_config(tmp_path / 'absent', {'storage': {'db_path': str(tmp_path / 'dream.db'), 'memory_root': str(memory)}, 'character': {'state_root': str(state), 'constitution_path': str(tmp_path / 'missing-constitution')}})
    conn = dream.open_db(cfg.db_path)
    insert_record(conn, 'character_moments', {'id': 'moment-one', 'project_slug': 'excluded-project'})
    insert_record(conn, 'character_pairs', {'id': 'pair-one', 'moment_id': 'moment-one', 'status': 'accepted', 'chosen': 'keep exact reviewed text'})
    insert_record(conn, 'character_judgments', {'id': 'judge-one', 'pair_id': 'pair-one'})
    insert_record(conn, 'character_review_events', {'id': 'event-one', 'pair_id': 'pair-one', 'reason': 'Human reason'})
    insert_record(conn, 'character_eval_runs', {'id': 'run-one', 'pair_ids': ['pair-one']})
    insert_record(conn, 'character_eval_results', {'id': 'result-one', 'run_id': 'run-one'})
    insert_record(conn, 'character_pairs', {'id': 'unrelated', 'status': 'pending', 'chosen': 'ordinary example'})
    conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime) VALUES('source','excluded-project','/fixture',0)")
    conn.execute("INSERT INTO messages(session_id,seq,role,text) VALUES('source',1,'user','original transcript')")
    conn.commit()
    (memory / '.dream-retention.json').write_text(json.dumps({'project_slugs': ['excluded-project']}))
    body = 'managed block'
    (state / 'block.md').write_text(body)
    (state / 'block.md.manifest.json').write_text(json.dumps({'pair_ids': ['pair-one'], 'files': {'block.md': hashlib.sha256(body.encode()).hexdigest()}}))
    yield conn, cfg
    conn.close()


def test_retention_backup_deletes_character_lineage_and_managed_files(setup):
    conn, cfg = setup
    preview = apply_retention(conn, cfg.memory_root, config=cfg)
    assert preview['character']['rows']['character_pairs'] == 1
    assert records(conn, 'character_pairs')[0]
    result = apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    assert result['character']['rerun_required'] is False
    assert [r['id'] for r in records(conn, 'character_pairs')] == ['unrelated']
    assert records(conn, 'character_review_events') == []
    assert records(conn, 'character_eval_results') == []
    assert not (Path(cfg.data['character']['state_root']) / 'block.md').exists()
    backup = json.loads((Path(result['backup']) / 'affected-records.json').read_text())
    assert backup['character']['rows']['character_review_events'][0]['reason'] == 'Human reason'
    assert 'block.md' in backup['character']['files']
    assert conn.execute('SELECT text FROM messages').fetchone()[0] == 'original transcript'
    assert apply_retention(conn, cfg.memory_root, config=cfg)['character']['rows']['character_pairs'] == 0


def test_character_snapshot_race_rolls_back_all_character_deletion(setup, monkeypatch):
    conn, cfg = setup
    real_dump = json.dump
    def race(*args, **kwargs):
        real_dump(*args, **kwargs)
        insert_record(conn, 'character_judgments', {'id': 'new-judge', 'pair_id': 'pair-one'})
        conn.commit()
    monkeypatch.setattr('retention_cleanup.json.dump', race)
    result = apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    assert result['character']['rerun_required'] is True
    assert len(records(conn, 'character_pairs')) == 2
    assert len(records(conn, 'character_judgments')) == 2
    assert (Path(cfg.data['character']['state_root']) / 'block.md').exists()


def test_user_modified_managed_file_is_preserved_and_reported(setup):
    conn, cfg = setup
    path = Path(cfg.data['character']['state_root']) / 'block.md'
    path.write_text('human changed this file')
    result = apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    assert path.read_text() == 'human changed this file'
    assert 'block.md' in result['character']['modified_files']


def test_character_policy_change_after_backup_aborts_character_part(setup, monkeypatch):
    conn, cfg = setup
    real_dump = json.dump
    def change_policy(*args, **kwargs):
        real_dump(*args, **kwargs)
        (cfg.memory_root / '.dream-retention.json').write_text('{}')
    monkeypatch.setattr('retention_cleanup.json.dump', change_policy)
    result = apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    assert result['character']['rerun_required'] is True
    assert len(records(conn, 'character_pairs')) == 2


def test_managed_artifact_symlink_cannot_delete_constitution(setup):
    conn, cfg = setup
    constitution = Path(cfg.data['character']['constitution_path'])
    constitution.write_text('human-owned document')
    path = Path(cfg.data['character']['state_root']) / 'block.md'
    path.unlink(); path.symlink_to(constitution)
    with pytest.raises(ValueError, match='symlink|protected'):
        apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    assert constitution.read_text() == 'human-owned document'
    assert len(records(conn, 'character_pairs')) == 2


def test_retention_command_reports_retry_as_nonzero(setup, monkeypatch, capsys):
    from types import SimpleNamespace
    conn, cfg = setup
    monkeypatch.setattr('character.retention.apply_snapshot', lambda *a: {'rerun_required': True, 'modified_files': []})
    assert dream.cmd_retention(SimpleNamespace(db=cfg.db_path, memory=str(cfg.memory_root), config=cfg, apply=True)) == 1
    assert 'Rerun' in capsys.readouterr().err


def test_retention_checks_managed_file_text_without_excluded_rows(setup):
    conn, cfg = setup
    (cfg.memory_root / '.dream-retention.json').write_text(json.dumps({'text_markers': ['managed block']}))
    preview = apply_retention(conn, cfg.memory_root, config=cfg)
    assert 'block.md' in preview['character']['managed_files']
    assert not any(preview['character']['rows'].values())
    result = apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    assert result['character']['rerun_required'] is False
    assert not (Path(cfg.data['character']['state_root']) / 'block.md').exists()
    assert len(records(conn, 'character_pairs')) == 2


def test_retention_removes_character_rows_from_listed_sessions(setup):
    conn, cfg = setup
    insert_record(conn, 'character_moments', {'id': 'moment-listed', 'session_id': 'listed'})
    insert_record(conn, 'character_pairs', {'id': 'pair-listed', 'moment_id': 'moment-listed', 'status': 'pending', 'chosen': 'listed example'})
    conn.commit()
    (cfg.memory_root / '.dream-retention.json').write_text(json.dumps({'session_ids': ['listed']}))
    apply_retention(conn, cfg.memory_root, config=cfg, apply=True)
    remaining = [r['id'] for r in records(conn, 'character_pairs')]
    assert 'pair-listed' not in remaining and 'unrelated' in remaining
