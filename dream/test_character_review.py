from test_character_judge import env,verdict
from types import SimpleNamespace
import pytest
from character import storage

def judged(env,monkeypatch):
    j,r,c,db,p=env; seq=iter([verdict(),verdict('B','violates','supports')])
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=next(seq)))
    j.judge_pairs(db,c,limit=1)
    return env

def test_accept_bound_and_edit_invalidates(env,monkeypatch):
    j,r,c,db,p=judged(env,monkeypatch)
    updated=r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked')
    assert updated['status']=='accepted' and r.accepted_pairs(db,c)
    edited=r.review_pair(db,c,'p','edit',expected_sha=updated['content_sha256'],body='New short',reason='Improve')
    assert edited['rejected']=='Long' and edited['status']=='edited'
    assert not r.accepted_pairs(db,c)
    with pytest.raises(ValueError): r.review_pair(db,c,'p','accept',expected_sha=edited['content_sha256'],reason='Checked')

def test_requires_judgment_and_current_hash(env):
    j,r,c,db,p=env
    with pytest.raises(ValueError): r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked')
    with pytest.raises(ValueError): r.review_pair(db,c,'p','defer',expected_sha='stale')

def test_flag_acknowledgment_and_reason(env,monkeypatch):
    j,r,c,db,p=env
    p=storage.update_record(db,'character_pairs',dict(p,revision=2,conflict=True),expected_sha=p['content_sha256']); db.commit()
    judged((j,r,c,db,p),monkeypatch)
    with pytest.raises(ValueError): r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked')
    updated=r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked',acknowledge_flags=True)
    assert updated['status']=='accepted'
    assert storage.records(db,'character_review_events')[0]['acknowledged_flags']==['conflict']

def test_rejected_cannot_accept_and_review_has_no_calls(env,monkeypatch):
    j,r,c,db,p=judged(env,monkeypatch)
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:pytest.fail('review provider call'))
    updated=r.review_pair(db,c,'p','reject',expected_sha=p['content_sha256'],reason='Reject')
    with pytest.raises(ValueError): r.review_pair(db,c,'p','accept',expected_sha=updated['content_sha256'],reason='Changed mind')
    assert r.show_pair(db,c,'p')['status']=='rejected'

def test_nested_policy_hides_pair_and_lineage(env):
    j,r,c,db,p=env
    c.memory_root.mkdir(); (c.memory_root/'.dream-retention.json').write_text('{"text_markers":["Long"]}')
    assert r.list_pairs(db,c)==[]
    with pytest.raises(ValueError): r.show_pair(db,c,'p')
    with pytest.raises(ValueError): r.review_pair(db,c,'p','defer',expected_sha=p['content_sha256'])

def test_edit_keeps_evidence_and_requires_new_judgment(env,monkeypatch):
    j,r,c,db,p=judged(env,monkeypatch)
    updated=r.review_pair(db,c,'p','edit',expected_sha=p['content_sha256'],reason='Better',body='Better short')
    assert updated['provenance']==p['provenance'] and updated['rejected']==p['rejected']
    assert storage.records(db,'character_judgments')
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('failed')))
    assert j.judge_pairs(db,c,limit=1)['failures']

def test_stale_history_visible_and_marked_on_mutation(env,monkeypatch):
    j,r,c,db,p=env
    older=storage.insert_record(db,'character_pairs',dict(p,id='old',constitution_sha256='old')); db.commit()
    assert r.show_pair(db,c,'old')['status']=='stale'
    r.review_pair(db,c,'p','defer',expected_sha=p['content_sha256'])
    assert storage.get_record(db,'character_pairs','old')['status']=='stale'

def test_accepted_decorations_do_not_change_hash(env,monkeypatch):
    j,r,c,db,p=judged(env,monkeypatch)
    accepted=r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Human checked')
    visible=r.accepted_pairs(db,c)[0]
    assert visible['acceptance_reason']=='Human checked' and visible['judgments']
    assert visible['content_sha256']==accepted['content_sha256']

def test_accept_right_exports_human_choice_and_keeps_original_evidence(env,monkeypatch):
    import json
    from character import export,render
    j,r,c,db,p=judged(env,monkeypatch)
    monkeypatch.setattr(export,'load_constitution',r.load_constitution)
    monkeypatch.setattr(render,'load_constitution',r.load_constitution)
    with pytest.raises(ValueError,match='acknowledgement'):
        r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Prefer right',answer='right')
    accepted=r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Prefer right',answer='right',acknowledge_flags=True)
    shown=r.show_pair(db,c,'p')
    assert shown['chosen']=='Short' and shown['rejected']=='Long'
    assert shown['accepted_answer']=='right'
    event=shown['review_events'][0]
    assert event['answer']=='right' and event['acknowledged_flags']==['against_selected']
    visible=r.accepted_pairs(db,c)[0]
    assert visible['chosen']=='Long' and visible['rejected']=='Short'
    assert visible['judgments'] and visible['content_sha256']==accepted['content_sha256']
    row=json.loads(export.export_pairs(db,c)['text'])
    assert row['chosen']=='Long' and row['rejected']=='Short'
    assert row['metadata']['review_event']['answer']=='right'
    # Full-pair acceptance alone must not inject an unreviewed short example.
    assert render.render_block(db,c)['manifest']['pair_ids']==[]
    edited=r.review_pair(db,c,'p','edit',expected_sha=accepted['content_sha256'],body='New answer',reason='Improve')
    assert 'accepted_answer' not in edited
    assert not r.accepted_pairs(db,c)

def test_accept_right_still_requires_current_hash_and_judgment(env,monkeypatch):
    j,r,c,db,p=env
    with pytest.raises(ValueError,match='judgment'):
        r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Right',answer='right',acknowledge_flags=True)
    judged(env,monkeypatch)
    with pytest.raises(ValueError,match='changed'):
        r.review_pair(db,c,'p','accept',expected_sha='stale',reason='Right',answer='right',acknowledge_flags=True)

def test_accept_right_with_matching_judge_needs_no_override(env,monkeypatch):
    j,r,c,db,p=env
    seq=iter([verdict('B','violates','supports'),verdict()])
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=next(seq)))
    assert j.judge_pairs(db,c,limit=1)['judged']==1
    r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Right',answer='right')
    assert r.accepted_pairs(db,c)[0]['chosen']=='Long'

def test_cli_accept_right_records_selection(env,monkeypatch,capsys):
    import argparse,json
    from character import cli
    j,r,c,db,p=judged(env,monkeypatch)
    parser=argparse.ArgumentParser()
    cli.register(parser.add_subparsers())
    args=parser.parse_args(['character','accept','p','--answer','right','--expected-sha',p['content_sha256'],'--reason','Right is better','--acknowledge-flags'])
    args.config=c
    args.db=c.db_path
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    assert cli.run(args)==0
    assert json.loads(capsys.readouterr().out)['accepted_answer']=='right'
    assert r.accepted_pairs(db,c)[0]['chosen']=='Long'

def test_draft_read_views_do_not_require_approval(env,monkeypatch):
    j,r,c,db,p=env; document=r.load_constitution(c)
    def load(config,*,require_approved=True):
        if require_approved: raise ValueError('draft')
        return document
    monkeypatch.setattr(r,'load_constitution',load)
    assert r.list_pairs(db,c)[0]['id']=='p'
    assert r.show_pair(db,c,'p')['id']=='p'
    with pytest.raises(ValueError): r.review_pair(db,c,'p','defer',expected_sha=p['content_sha256'])

def test_review_race_rechecks_live_row(env,monkeypatch):
    from contextlib import contextmanager
    j,r,c,db,p=env; original=storage.open_writer
    @contextmanager
    def raced(config):
        storage.update_record(db,'character_pairs',dict(p,revision=2,status='deferred'),expected_sha=p['content_sha256']); db.commit()
        with original(config) as writer: yield writer
    monkeypatch.setattr(storage,'open_writer',raced)
    with pytest.raises(ValueError): r.review_pair(db,c,'p','reject',expected_sha=p['content_sha256'],reason='Checked')
    assert not storage.records(db,'character_review_events')

@pytest.mark.parametrize('action',['accept','reject','defer','edit'])
def test_persisted_stale_cannot_revive(env,action):
    j,r,c,db,p=env
    p=storage.update_record(db,'character_pairs',dict(p,revision=2,status='stale'),expected_sha=p['content_sha256']); db.commit()
    with pytest.raises(ValueError):
        r.review_pair(db,c,'p',action,expected_sha=p['content_sha256'],reason='Checked',body='Edited')
    assert storage.get_record(db,'character_pairs','p')['status']=='stale'


def test_accept_rejects_changed_source_parent(env,monkeypatch):
    j,r,c,db,p=env
    parent=storage.insert_record(db,'character_seeds',dict(id='seed',revision=1,constitution_sha256='abc',seed={'prompt':'Original'}))
    p=storage.update_record(db,'character_pairs',dict(p,revision=2,seed_id='seed',provenance={'source_content_sha256':parent['content_sha256']}),expected_sha=p['content_sha256']); db.commit()
    judged((j,r,c,db,p),monkeypatch)
    storage.update_record(db,'character_seeds',dict(parent,revision=2,seed={'prompt':'Changed'}),expected_sha=parent['content_sha256']); db.commit()
    with pytest.raises(ValueError): r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked')

def test_stale_history_includes_old_judgments_and_reasons(env):
    j,r,c,db,p=env
    old=storage.insert_record(db,'character_pairs',dict(p,id='old',constitution_sha256='old'))
    storage.insert_record(db,'character_judgments',dict(id='old-judge',revision=1,pair_id='old',constitution_sha256='old',rationale='Historical verdict'))
    storage.insert_record(db,'character_review_events',dict(id='old-event',revision=1,pair_id='old',constitution_sha256='old',reason='Historical reason'))
    db.commit()
    shown=r.show_pair(db,c,'old')
    assert shown['status']=='stale'
    assert shown['judgments'][0]['rationale']=='Historical verdict'
    assert shown['review_events'][0]['reason']=='Historical reason'
