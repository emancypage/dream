"""Short examples require their own human review and keep source evidence intact."""
import json
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from test_character_judge import env
from test_character_review import judged
from character import storage, render, export, safety

BODY = {
    'situation': 'The user asks what proves a fix works.',
    'preferred': 'Run a check and report the result.',
    'avoid': 'Say it works without checking.',
    'feedback': 'The user wants evidence before a completion claim.',
    'limits': 'State which behavior the check covers.',
}


def prepared(env, monkeypatch):
    from character import short
    j, r, c, db, p = judged(env, monkeypatch)
    doc = r.load_constitution(c)
    for module in (short, render, export):
        monkeypatch.setattr(module, 'load_constitution', lambda *a, **k: doc)
    monkeypatch.setattr('character.constitution.load_constitution', lambda *a, **k: doc)
    pair = r.review_pair(db, c, 'p', 'accept', expected_sha=p['content_sha256'], reason='Checked')
    return short, r, c, db, pair


def approve_example(db, c, pair, body=None):
    """Test utility for explicit approval of a fixture short example."""
    from character import short
    draft = short.draft_example(db, c, pair['id'], expected_pair_sha=pair['content_sha256'], body=body or BODY)
    return short.review_example(db, c, draft['id'], 'approve', expected_sha=draft['content_sha256'], reason='Reviewed the short wording')


def test_draft_excluded_until_exact_human_approval_and_full_records_unchanged(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    before = storage.snapshot(db)
    exported = export.export_pairs(db, c)
    draft = short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], body=BODY)
    assert draft['status'] == 'draft'
    assert short.show_example(db, c, draft['id'])['body'] == BODY
    assert render.render_block(db, c)['manifest']['pair_ids'] == []
    with pytest.raises(ValueError, match='changed'):
        short.review_example(db, c, draft['id'], 'approve', expected_sha='stale', reason='Reviewed')
    with pytest.raises(ValueError, match='reason'):
        short.review_example(db, c, draft['id'], 'approve', expected_sha=draft['content_sha256'], reason='')
    approved = short.review_example(db, c, draft['id'], 'approve', expected_sha=draft['content_sha256'], reason='Reviewed')
    result = render.render_block(db, c)
    assert result['manifest']['pair_ids'] == ['p']
    assert result['manifest']['short_hashes'] == {approved['id']: approved['content_sha256']}
    assert BODY['situation'] in result['text'] and BODY['feedback'] in result['text']
    assert 'user: Task' not in result['text']
    assert export.export_pairs(db, c) == exported
    after = storage.snapshot(db)
    for table in ('character_pairs', 'character_moments', 'character_seeds', 'character_judgments'):
        assert after[table] == before[table]
    assert [e for e in after['character_review_events'] if e['action'] == 'accept'] == before['character_review_events']


def test_edit_invalidates_approval_and_old_preview(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    approved = approve_example(db, c, pair)
    preview = render.render_block(db, c)
    edited = short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], expected_sha=approved['content_sha256'], body=dict(BODY, preferred='Check again.'))
    assert edited['status'] == 'draft'
    assert not render.render_block(db, c)['manifest']['pair_ids']
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    with pytest.raises(ValueError, match='short example changed'):
        safety.write_artifacts(c, {'block.md': preview['text']}, preview['manifest'])
    with pytest.raises(ValueError, match='changed'):
        short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], expected_sha=approved['content_sha256'], body=BODY)


def test_source_pair_change_blocks_approval_and_render(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    approved = approve_example(db, c, pair)
    storage.update_record(db, 'character_pairs', dict(pair, revision=pair['revision']+1, status='rejected'), expected_sha=pair['content_sha256']); db.commit()
    assert not render.render_block(db, c)['manifest']['pair_ids']
    assert short.show_example(db, c, approved['id'])['status'] == 'stale'
    with pytest.raises(ValueError, match='source'):
        short.review_example(db, c, approved['id'], 'approve', expected_sha=approved['content_sha256'], reason='Reviewed')


def test_rejection_and_policy_exclusion_never_injected(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    approved = approve_example(db, c, pair)
    rejected = short.review_example(db, c, approved['id'], 'reject', expected_sha=approved['content_sha256'], reason='Wrong meaning')
    assert rejected['status'] == 'rejected'
    assert not render.render_block(db, c)['manifest']['pair_ids']
    c.memory_root.mkdir()
    (c.memory_root/'.dream-retention.json').write_text(json.dumps({'text_markers': [BODY['feedback']]}))
    assert short.list_examples(db, c) == []
    with pytest.raises(ValueError): short.show_example(db, c, approved['id'])


@pytest.mark.parametrize('body', [{}, dict(BODY, extra='unknown'), dict(BODY, feedback=''), dict(BODY, preferred='x'*2501), dict(BODY, limits=[])])
def test_invalid_short_body_rejected(env, monkeypatch, body):
    short, r, c, db, pair = prepared(env, monkeypatch)
    with pytest.raises(ValueError):
        short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], body=body)
    assert storage.records(db, 'character_short_examples') == []


def test_draft_needs_accepted_source_and_exact_source_hash(env, monkeypatch):
    from character import short
    j, r, c, db, pair = env
    monkeypatch.setattr(short, 'load_constitution', r.load_constitution)
    with pytest.raises(ValueError, match='source'):
        short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], body=BODY)
    short, r, c, db, pair = prepared(env, monkeypatch)
    with pytest.raises(ValueError, match='source'):
        short.draft_example(db, c, 'p', expected_pair_sha='stale', body=BODY)


def test_review_rechecks_source_under_writer_lock(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    draft = short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], body=BODY)
    original = storage.open_writer
    @contextmanager
    def race(config):
        storage.update_record(db, 'character_pairs', dict(pair, revision=pair['revision']+1, status='rejected'), expected_sha=pair['content_sha256']); db.commit()
        with original(config) as writer: yield writer
    monkeypatch.setattr(storage, 'open_writer', race)
    with pytest.raises(ValueError, match='source'):
        short.review_example(db, c, draft['id'], 'approve', expected_sha=draft['content_sha256'], reason='Reviewed')
    assert storage.get_record(db, 'character_short_examples', draft['id'])['status'] == 'draft'


def test_three_approved_short_examples_fit_budget(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    approve_example(db, c, pair)
    for identity in ('second', 'third'):
        # A copy is a fixture source with its own human review event.
        p = storage.insert_record(db, 'character_pairs', dict(pair, id=identity, revision=1, status='pending'))
        judgment = storage.records(db, 'character_judgments')[0]
        storage.insert_record(db, 'character_judgments', dict(judgment, id='judge-'+identity, pair_id=identity, pair_revision=1, pair_sha256=p['content_sha256']))
        db.commit()
        p = r.review_pair(db, c, identity, 'accept', expected_sha=p['content_sha256'], reason='Checked')
        approve_example(db, c, p)
    result = render.render_block(db, c)
    assert set(result['manifest']['pair_ids']) == {'p', 'second', 'third'}
    assert len(result['text']) <= c.data['character']['render_budget_codepoints']


def test_short_cli_draft_review_and_render_without_model_calls(env, monkeypatch, capsys, tmp_path):
    import argparse
    from character import cli
    short, r, c, db, pair = prepared(env, monkeypatch)
    monkeypatch.setattr('backend.generate', lambda *a, **k: pytest.fail('short commands must not call a provider'))
    body_file = tmp_path/'short.json'; body_file.write_text(json.dumps(BODY))
    def command(*argv):
        parser = argparse.ArgumentParser()
        cli.register(parser.add_subparsers())
        args = parser.parse_args(['character', *argv]); args.config = c; args.db = c.db_path
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        code = cli.run(args)
        captured = capsys.readouterr()
        assert code == 0, captured.err
        return json.loads(captured.out)
    draft = command('short', 'draft', 'p', '--expected-pair-sha', pair['content_sha256'], '--body-file', str(body_file))
    assert command('short', 'list')[0]['status'] == 'draft'
    assert command('short', 'show', draft['id'])['body'] == BODY
    assert command('render', '--preview')['manifest']['pair_ids'] == []
    approved = command('short', 'approve', draft['id'], '--expected-sha', draft['content_sha256'], '--reason', 'Approved this wording')
    assert approved['status'] == 'approved'
    command('render')
    assert BODY['preferred'] in (tmp_path/'state'/'block.md').read_text()
    command('short', 'reject', draft['id'], '--expected-sha', approved['content_sha256'], '--reason', 'Wrong meaning')
    assert command('render', '--preview')['manifest']['pair_ids'] == []


def test_retention_removes_short_example_and_its_review_by_lineage(env, monkeypatch):
    from character import retention
    short, r, c, db, pair = prepared(env, monkeypatch)
    approved = approve_example(db, c, pair)
    c.memory_root.mkdir()
    (c.memory_root/'.dream-retention.json').write_text(json.dumps({'text_markers':['Short']}))
    snapshot = retention.collect(db, c)
    assert snapshot['rows']['character_short_examples'][0]['id'] == approved['id']
    assert any(e['action'] == 'short_approve' for e in snapshot['rows']['character_review_events'])
    assert not retention.apply_snapshot(db, c, snapshot)['rerun_required']
    assert storage.records(db, 'character_short_examples') == []


def test_v3_upgrade_backs_up_full_records_before_short_table(tmp_path):
    import sqlite3
    import dream
    path = tmp_path/'db'
    db = dream.open_db(path)
    storage.insert_record(db, 'character_pairs', dict(id='source',chosen='Exact full text'))
    db.execute('DROP TABLE character_short_examples')
    db.execute('DELETE FROM schema_migrations WHERE version=4'); db.commit(); db.close()
    db = dream.open_db(path)
    assert storage.records(db, 'character_short_examples') == []
    assert storage.records(db, 'character_pairs')[0]['chosen'] == 'Exact full text'
    backup_path, = tmp_path.glob('db.bak-*')
    with sqlite3.connect(backup_path) as backup:
        assert backup.execute("SELECT name FROM sqlite_master WHERE name='character_short_examples'").fetchone() is None
        assert backup.execute('SELECT max(version) FROM schema_migrations').fetchone()[0] == 3
    db.close()


def test_forged_approval_without_human_event_is_not_injected(env, monkeypatch):
    short, r, c, db, pair = prepared(env, monkeypatch)
    draft = short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], body=BODY)
    storage.update_record(db, 'character_short_examples', dict(draft, revision=2, status='approved'), expected_sha=draft['content_sha256']); db.commit()
    assert not render.render_block(db, c)['manifest']['pair_ids']


def test_real_session_start_hook_uses_only_approved_short_text(env, monkeypatch, capsys):
    import io
    import dream
    from recall_types import RecallDiagnostics, RecallResult
    short, r, c, db, pair = prepared(env, monkeypatch)
    c.data['character']['hook_injection'] = True
    draft = short.draft_example(db, c, 'p', expected_pair_sha=pair['content_sha256'], body=BODY)
    monkeypatch.setattr(dream, 'claim_hook_event', lambda *a: 'token')
    monkeypatch.setattr(dream, 'finish_hook_event', lambda *a, **k: None)
    monkeypatch.setattr(dream, 'run_context', lambda *a, **k: RecallResult((), '', 'lexical', None, RecallDiagnostics(0, 0, None, 0, 0, None)))
    def hook():
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        monkeypatch.setattr(dream.sys, 'stdin', io.StringIO(json.dumps({'session_id':'short-hook','source':'startup','cwd':str(c.db_path.parent)})))
        assert dream.cmd_context(SimpleNamespace(context_cmd='session-start', config=c, db=c.db_path, explain=False)) == 0
        return json.loads(capsys.readouterr().out)['hookSpecificOutput']['additionalContext']
    assert BODY['feedback'] not in hook()
    short.review_example(db, c, draft['id'], 'approve', expected_sha=draft['content_sha256'], reason='Approved exact wording')
    text = hook()
    assert BODY['feedback'] in text and BODY['limits'] in text
    assert 'user: Task' not in text
