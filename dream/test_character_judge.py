import sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from config import load_config
from character import storage

@pytest.fixture
def env(tmp_path, monkeypatch):
    from character import judge, review
    cfg=load_config(tmp_path/'none', {'storage':{'db_path':str(tmp_path/'db'),'memory_root':str(tmp_path/'memory')},'character':{'enabled':True,'constitution_path':str(tmp_path/'constitution'),'state_root':str(tmp_path/'state')}})
    doc=SimpleNamespace(text='Rules',sha256='abc',version='1',traits={'brief':'Be brief'})
    for mod in (judge,review): monkeypatch.setattr(mod,'load_constitution',lambda *a,**k:doc)
    conn=sqlite3.connect(cfg.db_path); conn.executescript((Path(__file__).parent/'schema.sql').read_text())
    pair=storage.insert_record(conn,'character_pairs',dict(id='p',revision=1,constitution_sha256='abc',prompt_messages=[{'role':'user','content':'Task'}],chosen='Short',rejected='Long',trait_ids=['brief'],conflict=False,status='pending',provenance={'historical_family':'anthropic'},generator_calls=[])); conn.commit()
    return judge,review,cfg,conn,pair

def verdict(winner='A',a='supports',b='violates'):
    return {'winner':winner,'assessments':[{'trait_id':'brief','a_assessment':a,'b_assessment':b,'rationale':'Evidence'}],'rationale':'Better'}

def test_swapped_normalization(env,monkeypatch):
    j,r,c,db,p=env; calls=[]
    def fake(*args,**kw):
        calls.append(args[1]); return SimpleNamespace(output=verdict() if len(calls)==1 else verdict('B','violates','supports'))
    monkeypatch.setattr(j.backend,'generate',fake)
    assert j.judge_pairs(db,c,limit=1)['judged']==1
    row=storage.records(db,'character_judgments')[0]
    assert row['normalized_result']=='chosen' and len(row['calls'])==2
    assert 'chosen' not in calls[0] and 'generator' not in calls[0]
    assert storage.get_record(db,'character_pairs','p')['status']=='pending'

def test_failed_second_retryable(env,monkeypatch):
    j,r,c,db,p=env; answers=iter([SimpleNamespace(output=verdict()),RuntimeError('secret')])
    def fake(*a,**k):
        v=next(answers)
        if isinstance(v,Exception): raise v
        return v
    monkeypatch.setattr(j.backend,'generate',fake)
    assert j.judge_pairs(db,c,limit=1)['failures']
    assert not storage.records(db,'character_judgments')

def test_bad_assessment_rejected(env,monkeypatch):
    j,r,c,db,p=env
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=verdict(a='excellent')))
    assert j.judge_pairs(db,c,limit=1)['failures']

@pytest.mark.parametrize('second,expected',[(verdict('B','violates','supports'),'chosen'),(verdict('B','unclear','supports'),'inconsistent'),(verdict('tie','supports','violates'),'inconsistent')])
def test_assessment_disagreement(env,monkeypatch,second,expected):
    j,r,c,db,p=env; seq=iter([verdict(),second])
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=next(seq)))
    assert j.judge_pairs(db,c,limit=1)['judged']==1
    assert storage.records(db,'character_judgments')[0]['normalized_result']==expected

@pytest.mark.parametrize('winner',['tie','neither'])
def test_stable_nonwinner(env,monkeypatch,winner):
    j,r,c,db,p=env; seq=iter([verdict(winner),verdict(winner,'violates','supports')])
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=next(seq)))
    assert j.judge_pairs(db,c,limit=1)['judged']==1
    assert storage.records(db,'character_judgments')[0]['normalized_result']==winner

def test_mandatory_family_check_and_explicit_override(env,monkeypatch):
    j,r,c,db,p=env
    monkeypatch.setattr(j,'provider_family',lambda *a:'anthropic')
    with pytest.raises(ValueError): j.judge_pairs(db,c,limit=1)
    c.data['character']['allow_same_family_judge']=True
    seq=iter([verdict(),verdict('B','violates','supports')])
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=next(seq)))
    assert j.judge_pairs(db,c,limit=1)['judged']==1
    row=storage.records(db,'character_judgments')[0]
    assert row['same_family_override'] and row['historical_overlap_warning']
    assert all(call['same_family_override'] for call in row['calls'])
    assert r.show_pair(db,c,'p')['historical_overlap_warning']

def test_policy_race_no_completed_result(env,monkeypatch):
    j,r,c,db,p=env
    def fake(*a,**k):
        c.memory_root.mkdir(); (c.memory_root/'.dream-retention.json').write_text('{"instructions":"changed"}')
        return SimpleNamespace(output=verdict())
    monkeypatch.setattr(j.backend,'generate',fake)
    assert j.judge_pairs(db,c,limit=1)['failures']
    assert not storage.records(db,'character_judgments')

def test_unknown_auth_rejected_before_calls(env,monkeypatch):
    j,r,c,db,p=env
    route=c.stage('character_judge'); c.data['providers'][route['provider']]['auth']='unknown'
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:pytest.fail('must not call'))
    with pytest.raises(ValueError): j.judge_pairs(db,c,limit=1)

def test_readonly_input_and_retry_skip(env,monkeypatch):
    j,r,c,db,p=env; seq=iter([verdict(),verdict('B','violates','supports')])
    monkeypatch.setattr(j.backend,'generate',lambda *a,**k:SimpleNamespace(output=next(seq)))
    readonly=sqlite3.connect('file:'+str(c.db_path)+'?mode=ro',uri=True)
    assert j.judge_pairs(readonly,c,limit=1)['judged']==1
    assert j.judge_pairs(readonly,c,limit=1)['judged']==0
    readonly.close()

def test_pair_race_discards_both_calls(env,monkeypatch):
    j,r,c,db,p=env
    def fake(*a,**k):
        storage.update_record(db,'character_pairs',dict(p,revision=2,chosen='Changed'),expected_sha=p['content_sha256']); db.commit()
        return SimpleNamespace(output=verdict())
    monkeypatch.setattr(j.backend,'generate',fake)
    assert j.judge_pairs(db,c,limit=1)['failures']
    assert not storage.records(db,'character_judgments')
