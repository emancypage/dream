"""Acceptance drafts wording, while only a later human review can activate it."""
import argparse
import json
import sqlite3
from types import SimpleNamespace
import pytest
from test_character_judge import env
from test_character_review import judged
from test_character_short import BODY, prepared, approve_example
from character import cli, short, storage, render


def command(c, db, capsys, *argv):
    parser = argparse.ArgumentParser()
    cli.register(parser.add_subparsers())
    args = parser.parse_args(['character', *argv]); args.config = c; args.db = c.db_path
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    if argv[0] == 'accept':
        db.close()  # Match a CLI process without a fixture writer holding its WAL open.
    code = cli.run(args)
    result = capsys.readouterr()
    assert result.out, result.err
    return code, json.loads(result.out)


@pytest.mark.parametrize('answer,preferred,avoided', [('left','Short','Long'), ('right','Long','Short')])
def test_cli_accept_automatically_drafts_selected_answer_pending_review(env, monkeypatch, capsys, answer, preferred, avoided):
    j, r, c, db, p = judged(env, monkeypatch)
    c.data['stages']['character_generate']['model'] = 'configured-generator'
    monkeypatch.setattr(render, 'load_constitution', r.load_constitution)
    def generate(stage, prompt, schema, **kwargs):
        assert stage == 'character_generate' and kwargs['config'].stage(stage)['model'] == 'configured-generator'
        payload = json.loads(prompt)
        assert payload['preferred_answer'] == preferred and payload['avoided_answer'] == avoided
        assert payload['human_feedback'] == 'Human selected this answer'
        assert payload['messages'] == [{'role':'user','content':'Task'}]
        return SimpleNamespace(output=BODY, provider='fake', model='configured-generator', duration_ms=1)
    monkeypatch.setattr(j.backend, 'generate', generate)
    code, result = command(c, db, capsys, 'accept', 'p', '--answer', answer, '--expected-sha', p['content_sha256'], '--reason', 'Human selected this answer', '--acknowledge-flags')
    db = sqlite3.connect(c.db_path)
    assert code == 0 and result['status'] == 'accepted'
    draft = result['short_draft']
    assert draft['body'] == BODY and draft['status'] == 'draft'
    assert draft['pair_sha256'] == result['content_sha256']
    assert draft['generation']['configured_model'] == 'configured-generator'
    assert not render.render_block(db,c)['manifest']['pair_ids']
    code, approved = command(c, db, capsys, 'short', 'approve', draft['id'], '--expected-sha', draft['content_sha256'], '--reason', 'Approved the short wording')
    assert code == 0 and approved['status'] == 'approved'
    assert render.render_block(db,c)['manifest']['pair_ids'] == ['p']


@pytest.mark.parametrize('failure', ['provider','schema','length'])
def test_failed_auto_draft_preserves_acceptance_and_can_retry(env, monkeypatch, capsys, failure):
    j, r, c, db, p = judged(env, monkeypatch)
    def fail(*a, **k):
        if failure == 'provider': raise RuntimeError('private provider message')
        return SimpleNamespace(output={} if failure == 'schema' else dict(BODY,preferred='x'*2501))
    monkeypatch.setattr(j.backend, 'generate', fail)
    code, result = command(c,db,capsys,'accept','p','--expected-sha',p['content_sha256'],'--reason','Checked')
    db = sqlite3.connect(c.db_path)
    assert code == 1 and result['status'] == 'accepted' and result['failures']
    assert 'private provider message' not in json.dumps(result)
    assert r.show_pair(db,c,'p')['status'] == 'accepted'
    assert storage.records(db,short.TABLE) == []
    monkeypatch.setattr(j.backend, 'generate', lambda *a,**k:SimpleNamespace(output=BODY))
    code, draft = command(c,db,capsys,'short','generate','p','--expected-pair-sha',result['content_sha256'])
    assert code == 0 and draft['status'] == 'draft'
    assert r.show_pair(db,c,'p')['content_sha256'] == result['content_sha256']


def test_existing_human_draft_is_not_overwritten(env, monkeypatch, capsys):
    short_module, r, c, db, pair = prepared(env,monkeypatch)
    approved = approve_example(db,c,pair)
    monkeypatch.setattr('backend.generate', lambda *a,**k:pytest.fail('existing wording must not trigger a call'))
    code, result = command(c,db,capsys,'short','generate','p','--expected-pair-sha',pair['content_sha256'])
    assert code == 0 and result['skipped'] == 'short example already exists for this source'
    assert storage.get_record(db,short.TABLE,approved['id']) == approved


def test_source_change_during_generation_aborts_draft(env, monkeypatch, capsys):
    short_module, r, c, db, pair = prepared(env,monkeypatch)
    def generate(*a,**k):
        storage.update_record(db,'character_pairs',dict(pair,revision=pair['revision']+1,status='rejected'),expected_sha=pair['content_sha256']);db.commit()
        return SimpleNamespace(output=BODY)
    monkeypatch.setattr('backend.generate',generate)
    code, result = command(c,db,capsys,'short','generate','p','--expected-pair-sha',pair['content_sha256'])
    assert code == 1 and result['failures']
    assert storage.records(db,short.TABLE) == []


def test_disabled_generation_accepts_without_model_calls(env, monkeypatch, capsys):
    j,r,c,db,p = judged(env,monkeypatch)
    c.data['character']['enabled'] = False
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:pytest.fail('disabled generation'))
    code,result = command(c,db,capsys,'accept','p','--expected-sha',p['content_sha256'],'--reason','Checked')
    db = sqlite3.connect(c.db_path)
    assert code == 0 and result['status'] == 'accepted'
    assert result['short_draft']['skipped'] == 'character disabled'
    assert storage.records(db,short.TABLE) == []


def test_policy_change_during_generation_aborts_draft(env, monkeypatch, capsys):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    def generate(*a,**k):
        c.memory_root.mkdir()
        (c.memory_root/'.dream-retention.json').write_text('{"text_markers":["new-exclusion"]}')
        return SimpleNamespace(output=BODY)
    monkeypatch.setattr('backend.generate',generate)
    code,result = command(c,db,capsys,'short','generate','p','--expected-pair-sha',pair['content_sha256'])
    assert code == 1 and result['failures']
    assert storage.records(db,short.TABLE) == []


def test_renewed_full_acceptance_replaces_stale_short_with_unapproved_draft(env, monkeypatch, capsys):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    approved = approve_example(db,c,pair)
    from character import judge
    pair = r.review_pair(db,c,'p','edit',expected_sha=pair['content_sha256'],body='Updated full answer',reason='Corrected full wording')
    judged((judge,r,c,db,pair),monkeypatch)
    pair = r.review_pair(db,c,'p','accept',expected_sha=pair['content_sha256'],reason='Reviewed again')
    monkeypatch.setattr('backend.generate',lambda *a,**k:SimpleNamespace(output=dict(BODY,preferred='Report the exact check result.')))
    code,draft = command(c,db,capsys,'short','generate','p','--expected-pair-sha',pair['content_sha256'])
    assert code == 0 and draft['id'] == approved['id'] and draft['status'] == 'draft'
    assert draft['pair_sha256'] == pair['content_sha256']
    assert not render.render_block(db,c)['manifest']['pair_ids']


def test_human_draft_created_during_model_call_is_preserved(env, monkeypatch, capsys):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    manual_body = dict(BODY, preferred='Human wrote this exact text.')
    def generate(*a,**k):
        short.draft_example(db,c,'p',expected_pair_sha=pair['content_sha256'],body=manual_body)
        return SimpleNamespace(output=BODY)
    monkeypatch.setattr('backend.generate',generate)
    code,result = command(c,db,capsys,'short','generate','p','--expected-pair-sha',pair['content_sha256'])
    assert code == 1 and result['failures']
    assert storage.records(db,short.TABLE)[0]['body'] == manual_body


def test_read_failure_after_acceptance_still_reports_saved_human_choice(env, monkeypatch, capsys):
    j,r,c,db,p = judged(env,monkeypatch)
    original_review = r.review_pair
    original_reader = cli.open_db_readonly
    saved = []
    def review(*a,**k):
        result = original_review(*a,**k)
        saved.append(result)
        return result
    def reader(path):
        if saved: raise sqlite3.OperationalError('active WAL')
        return original_reader(path)
    monkeypatch.setattr(r,'review_pair',review)
    monkeypatch.setattr(cli,'open_db_readonly',reader)
    code,result = command(c,db,capsys,'accept','p','--expected-sha',p['content_sha256'],'--reason','Checked')
    assert code == 1 and result['status'] == 'accepted' and result['short_draft']['failures']
    db = sqlite3.connect(c.db_path)
    assert r.show_pair(db,c,'p')['status'] == 'accepted'
