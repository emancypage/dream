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
