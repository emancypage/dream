import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from config import load_config
from character import storage
from character.types import content_hash

@pytest.fixture
def env(tmp_path,monkeypatch):
    from character import eval as ev
    cfg=load_config(tmp_path/'none',{'storage':{'db_path':str(tmp_path/'db'),'memory_root':str(tmp_path/'memory')},'character':{'enabled':True,'constitution_path':str(tmp_path/'constitution'),'state_root':str(tmp_path/'state')}})
    for stage in ('character_eval_response','character_judge'): cfg.data['stages'][stage]['model']='fake-model'
    doc=SimpleNamespace(text='Rules',sha256='abc',version='1',traits={'brief':'Brief. Why: useful'})
    monkeypatch.setattr(ev,'load_constitution',lambda *a:doc)
    monkeypatch.setattr(ev,'render_block',lambda *a:{'text':'Rules','manifest':{'constitution_sha256':'abc','constitution_version':'1','pair_ids':[],'pair_hashes':{},'pair_revisions':{},'policy_fingerprint':ev.policy_fingerprint(ev.policy_for(cfg)),'content_sha256':'render','historical_overlap_warning':True,'same_family_override':False}})
    monkeypatch.setattr(ev,'provider_family',lambda c,s:'judge' if s=='character_judge' else 'response')
    db=sqlite3.connect(cfg.db_path); db.executescript((Path(__file__).parent/'schema.sql').read_text()); db.commit(); db.close(); db=sqlite3.connect('file:'+str(cfg.db_path)+'?immutable=1',uri=True)
    fixture=tmp_path/'fixtures.json'
    probe={'id':'pressure','category':'pressure','messages':[{'role':'user','content':'First'},{'role':'user','content':'Again'}],'trait_ids':['brief'],'trait_sha256':{'brief':content_hash(doc.traits['brief'])},'rubric':{'usefulness':{str(i):str(i) for i in range(5)}}}
    fixture.write_text(json.dumps({'version':'v1','probes':[probe]}))
    return ev,cfg,db,fixture,probe

def report(ev,cfg,run_id):
    fresh=ev.open_db_readonly(cfg.db_path)
    try: return ev.report_run(fresh,cfg,run_id)
    finally: fresh.close()

def fake_scores(swapped=False,score=3):
    return {'scores':{'A':{'usefulness':score},'B':{'usefulness':score}},'rationale':'Both answers are useful.'}

def test_multiturn_and_report(env,monkeypatch):
    ev,cfg,db,f,p=env; calls=[]
    def fake(stage,prompt,schema,**kw):
        calls.append((stage,json.loads(prompt),kw['system_prompt']))
        return SimpleNamespace(output=fake_scores() if stage=='character_judge' else {'response':'Answer'})
    monkeypatch.setattr(ev.backend,'generate',fake)
    result=ev.evaluate(db,cfg,fixtures=f,limit=1)
    assert not result['failures']
    assert len(calls)==6
    assert calls[1][1]['messages'][1]=={'role':'assistant','content':'Answer'}
    assert calls[0][2]!=calls[2][2]
    summary=report(ev,cfg,result['run_id'])
    assert summary['categories'][0]['valid']==1
    assert summary['categories'][0]['delta']==0
    assert summary['manifest']['historical_overlap_warning']
    assert summary['markdown']

def test_unsupported_without_calls(env,monkeypatch):
    ev,c,db,f,p=env; p['trait_sha256']['brief']='wrong'; f.write_text(json.dumps({'version':'v1','probes':[p]}))
    monkeypatch.setattr(ev.backend,'generate',lambda *a,**k:pytest.fail('unsupported must not call'))
    result=ev.evaluate(db,c,fixtures=f,limit=1)
    assert report(ev,c,result['run_id'])['categories'][0]['unsupported']==1

@pytest.mark.parametrize('mode',['second_failure','range','inconsistent','partial'])
def test_failure_states(env,monkeypatch,mode):
    ev,c,db,f,p=env; n=0
    def fake(stage,*args,**kw):
        nonlocal n
        n+=1
        if mode=='partial' and n==3: raise RuntimeError('private raw text')
        if stage!='character_judge': return SimpleNamespace(output={'response':'Answer'})
        if mode=='second_failure' and n==6: raise RuntimeError('private raw text')
        return SimpleNamespace(output=fake_scores(score=5 if mode=='range' else 2 if mode=='inconsistent' and n==6 else 3))
    monkeypatch.setattr(ev.backend,'generate',fake)
    run=ev.evaluate(db,c,fixtures=f,limit=1); summary=report(ev,c,run['run_id'])
    assert summary['categories'][0]['valid']==0
    fresh=ev.open_db_readonly(c.db_path); rows=storage.records(fresh,'character_eval_results'); fresh.close()
    assert any(r['state']=='generated' for r in rows) or any(r.get('responses') for r in rows)
    if mode!='inconsistent': assert run['failures']
    assert 'private raw text' not in json.dumps(rows)

@pytest.mark.parametrize('bad',[True,-1,5,1.5,'3'])
def test_strict_integer_rubric(env,monkeypatch,bad):
    ev,c,db,f,p=env
    monkeypatch.setattr(ev.backend,'generate',lambda stage,*a,**kw:SimpleNamespace(output=fake_scores(score=bad) if stage=='character_judge' else {'response':'Answer'}))
    result=ev.evaluate(db,c,fixtures=f,limit=1)
    assert result['failures']
    assert report(ev,c,result['run_id'])['categories'][0]['valid']==0


def test_swapped_anonymous_different_scores(env,monkeypatch):
    ev,c,db,f,p=env; judges=[]
    def fake(stage,prompt,*a,**kw):
        if stage!='character_judge': return SimpleNamespace(output={'response':'Reply'})
        judges.append(json.loads(prompt)); swapped=len(judges)==2
        return SimpleNamespace(output={'scores':{'A':{'usefulness':4 if swapped else 1},'B':{'usefulness':1 if swapped else 4}},'rationale':'The second trajectory is more useful.'})
    monkeypatch.setattr(ev.backend,'generate',fake)
    result=ev.evaluate(db,c,fixtures=f,limit=1); summary=report(ev,c,result['run_id'])
    assert summary['categories'][0]['delta']==3
    assert set(judges[0])=={'rubric','constitution','A','B'}
    calls=summary['results'][0]['calls']
    assert len(calls)==4
    assert all(set(('started_at','finished_at','provider','auth_family','configured_model','reported_model','prompt_sha256','usage','duration_ms')) <= call.keys() for call in calls)


def test_family_guard_before_calls(env,monkeypatch):
    ev,c,db,f,p=env
    monkeypatch.setattr(ev,'provider_family',lambda *a:'same')
    monkeypatch.setattr(ev.backend,'generate',lambda *a,**k:pytest.fail('family collision'))
    with pytest.raises(ValueError,match='independent'): ev.evaluate(db,c,fixtures=f,limit=1)


def test_policy_change_during_response_discards(env,monkeypatch):
    ev,c,db,f,p=env; calls=[]
    def fake(*a,**kw):
        calls.append(1); c.memory_root.mkdir(exist_ok=True)
        (c.memory_root/'.dream-retention.json').write_text('{"instructions":"Changed"}')
        return SimpleNamespace(output={'response':'Answer'})
    monkeypatch.setattr(ev.backend,'generate',fake)
    result=ev.evaluate(db,c,fixtures=f,limit=1)
    assert result['failures'] and len(calls)==1
    fresh=ev.open_db_readonly(c.db_path)
    assert not storage.records(fresh,'character_eval_results'); fresh.close()


def test_report_readonly_and_excluded_render_dependency(env,monkeypatch):
    ev,c,db,f,p=env
    with storage.open_writer(c) as writer:
        storage.insert_record(writer,'character_pairs',{'id':'accepted','constitution_sha256':'abc','chosen':'secretmarker'})
    rendered=ev.render_block(db,c); rendered['manifest']['pair_ids']=['accepted']
    monkeypatch.setattr(ev,'render_block',lambda *a:rendered)
    monkeypatch.setattr(ev.backend,'generate',lambda stage,*a,**kw:SimpleNamespace(output=fake_scores() if stage=='character_judge' else {'response':'Answer'}))
    run=ev.evaluate(db,c,fixtures=f,limit=1)
    monkeypatch.setattr(storage,'open_writer',lambda *a:pytest.fail('report must be read only'))
    monkeypatch.setattr(ev.backend,'generate',lambda *a,**kw:pytest.fail('report must not call providers'))
    assert report(ev,c,run['run_id'])['categories'][0]['valid']==1
    c.memory_root.mkdir(exist_ok=True); (c.memory_root/'.dream-retention.json').write_text('{"text_markers":["secretmarker"]}')
    with pytest.raises(ValueError,match='unavailable'): report(ev,c,run['run_id'])


def test_excluded_probe_roster_before_limit(env,monkeypatch):
    ev,c,db,f,p=env
    excluded=dict(p,id='private-marker',category='private-marker')
    f.write_text(json.dumps({'version':'v1','probes':[excluded,p]}))
    c.memory_root.mkdir(); (c.memory_root/'.dream-retention.json').write_text('{"text_markers":["private-marker"]}')
    calls=[]
    def fake(stage,*a,**kw):
        calls.append(stage); return SimpleNamespace(output=fake_scores() if stage=='character_judge' else {'response':'Answer'})
    monkeypatch.setattr(ev.backend,'generate',fake)
    run=ev.evaluate(db,c,fixtures=f,limit=1)
    summary=report(ev,c,run['run_id'])
    assert summary['categories'][0]['valid']==1 and len(calls)==6
    fresh=ev.open_db_readonly(c.db_path)
    assert 'private-marker' not in json.dumps(storage.snapshot(fresh)); fresh.close()
    assert summary['policy_skipped']==1


def test_bounded_context_isolated(env,monkeypatch):
    ev,c,db,f,p=env; c.data['character']['max_context_codepoints']=2
    monkeypatch.setattr(ev.backend,'generate',lambda *a,**kw:pytest.fail('oversized context'))
    run=ev.evaluate(db,c,fixtures=f,limit=1)
    assert run['failures']


def test_common_subset_controls_and_shared_provenance(env,monkeypatch):
    ev,c,db,f,p=env
    control=dict(p,id='neutral',category='task-control',trait_ids=[],trait_sha256={})
    broken=dict(p,id='broken',category='task-control',trait_ids=[],trait_sha256={})
    f.write_text(json.dumps({'version':'v1','probes':[control,broken]})); count=0
    def fake(stage,*a,**kw):
        nonlocal count
        count+=1
        if count==9: raise RuntimeError('provider detail')
        return SimpleNamespace(output=fake_scores() if stage=='character_judge' else {'response':'Answer'})
    monkeypatch.setattr(ev.backend,'generate',fake)
    run=ev.evaluate(db,c,fixtures=f,limit=2); summary=report(ev,c,run['run_id'])
    category=summary['categories'][0]
    assert category['total']==2 and category['valid']==1 and category['failed']==1
    assert category['dimensions']['usefulness']['valid']==1
    assert all(set(('constitution_version','prompt_version','created_at')) <= row.keys() for row in summary['results'])


def test_response_bound_rejects_fake_output(env,monkeypatch):
    ev,c,db,f,p=env; c.data['character']['max_response_codepoints']=3
    monkeypatch.setattr(ev.backend,'generate',lambda *a,**kw:SimpleNamespace(output={'response':'Too long'}))
    run=ev.evaluate(db,c,fixtures=f,limit=1)
    assert run['failures']
    assert all(not r.get('responses') for r in report(ev,c,run['run_id'])['results'])


@pytest.mark.parametrize('mode', ['scored', 'unsupported', 'failed'])
def test_late_policy_hides_probe_rubric_dependencies(env, monkeypatch, mode):
    ev, cfg, db, fixture, probe = env
    probe['rubric']['usefulness']['0'] = 'private-topic-marker'
    if mode == 'unsupported': probe['trait_sha256']['brief'] = 'wrong'
    fixture.write_text(json.dumps({'version': 'v1', 'probes': [probe]}))
    def fake(stage, *a, **k):
        if mode == 'failed': raise RuntimeError('provider unavailable')
        return SimpleNamespace(output=fake_scores() if stage == 'character_judge' else {'response': 'Answer'})
    monkeypatch.setattr(ev.backend, 'generate', fake)
    result = ev.evaluate(db, cfg, fixtures=fixture, limit=1)
    cfg.memory_root.mkdir()
    (cfg.memory_root / '.dream-retention.json').write_text(json.dumps({'text_markers': ['private-topic-marker']}))
    summary = report(ev, cfg, result['run_id'])
    assert summary['categories'][0]['valid'] == 0
    assert summary['results'] == []


def test_report_manifest_carries_probe_result_lineage(env, monkeypatch):
    ev, cfg, db, fixture, probe = env
    monkeypatch.setattr(ev.backend, 'generate', lambda stage, *a, **k: SimpleNamespace(output=fake_scores() if stage == 'character_judge' else {'response': 'Answer'}))
    result = ev.evaluate(db, cfg, fixtures=fixture, limit=1)
    summary = report(ev, cfg, result['run_id'])
    assert set(summary['manifest']['result_ids']) == {row['id'] for row in summary['results']}
