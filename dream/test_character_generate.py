import sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from config import load_config
from character import storage

@pytest.fixture
def env(tmp_path, monkeypatch):
    from character import generate as g
    cfg = load_config(tmp_path/'none', {'storage': {'db_path': str(tmp_path/'db'), 'memory_root': str(tmp_path/'memory')}, 'character': {'enabled': True, 'constitution_path': str(tmp_path/'constitution'), 'state_root': str(tmp_path/'state')}})
    doc = SimpleNamespace(text='Approved rules', sha256='abc', version='1', traits={'trait-a':'Be brief. Why: clarity'}, seeds=())
    monkeypatch.setattr(g, 'load_constitution', lambda *a, **k: doc)
    conn=sqlite3.connect(cfg.db_path)
    conn.executescript((Path(__file__).parent/'schema.sql').read_text())
    conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime,source_revision,parser_version) VALUES ('s','project','/source',1,'r1','v1')")
    conn.executemany('INSERT INTO messages(session_id,seq,role,text) VALUES (?,?,?,?)', [('s',1,'user','Explain it'),('s',2,'assistant','Long answer'),('s',3,'user','Nie tak, shorter please')])
    conn.commit()
    return g,cfg,conn

def extraction(**changes):
    return dict(is_correction=True,context_excerpt='Explain it',corrected_assistant_response='Long answer',user_correction='Nie tak, shorter please',trait_ids=['trait-a'],agreement='agrees',context_sufficient=True,rationale='Too long',**changes)

def test_estimate_and_correction_no_leak(env,monkeypatch):
    g,cfg,conn=env
    calls=[]
    def fake(stage,prompt,schema,**kw):
        calls.append((stage,prompt,kw))
        return SimpleNamespace(output=extraction() if stage=='character_extract' else {'response':'Brief answer'})
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.estimate(conn,cfg,source='all',limit=1)['source_items']==1
    assert calls==[]
    assert g.generate_pairs(conn,cfg,source='all',limit=1)['generated']==1
    assert 'Nie tak' not in calls[1][1] and 'Too long' not in calls[1][1]
    pair=storage.records(conn,'character_pairs')[0]
    assert pair['provenance']['historical_family']=='anthropic'
    assert pair['status']=='pending'
    assert g.generate_pairs(conn,cfg,source='all',limit=1)['generated']==0
    assert g.generate_pairs(conn,cfg,source='all',limit=1,refresh=True)['generated']==1
    assert len(storage.records(conn,'character_pairs'))==2

def test_negative_cached_and_failure_retry(env,monkeypatch):
    g,cfg,conn=env
    bad=extraction(); bad['is_correction']=False
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: SimpleNamespace(output=bad))
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['generated']==0
    assert g.estimate(conn,cfg,source='corrections',limit=1)['source_items']==0

def test_schema_failures_are_safe(env,monkeypatch):
    g,cfg,conn=env
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: SimpleNamespace(output={'secret':'private text'}))
    result=g.generate_pairs(conn,cfg,source='corrections',limit=1)
    assert len(result['failures'])==1 and 'private text' not in str(result)
    assert storage.records(conn,'character_moments')==[]

def test_seedless_and_oversized_prelimit(env):
    g,cfg,conn=env
    assert g.estimate(conn,cfg,source='synthetic',limit=2)['source_items']==0
    cfg.data['character']['max_context_codepoints']=2
    result=g.estimate(conn,cfg,source='all',limit=1)
    assert result['source_items']==0 and result['skipped']['oversized']==1

def test_synthetic_isolation_and_equal_failure(env,monkeypatch):
    g,cfg,conn=env
    doc=g.load_constitution(cfg)
    doc.seeds=({'id':'seed-a','trait_id':'trait-a','prompt':'Explain a task'},)
    calls=[]
    def fake(stage,prompt,schema,**kw):
        calls.append((prompt,kw['system_prompt']))
        return SimpleNamespace(output={'response':'chosen secret' if len(calls)%2 else 'baseline'})
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1)['generated']==1
    assert 'chosen secret' not in calls[1][0]
    assert doc.text in calls[0][1] and doc.text not in calls[1][1]

def test_policy_race_discards_and_sql_errors_propagate(env,monkeypatch):
    g,cfg,conn=env
    def fake(*a,**k):
        cfg.memory_root.mkdir()
        (cfg.memory_root/'.dream-retention.json').write_text('{"instructions":"new"}')
        return SimpleNamespace(output=extraction())
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['failures']
    assert storage.records(conn,'character_moments')==[]
    conn.close()
    with pytest.raises(sqlite3.Error):
        g.estimate(conn,cfg,source='corrections',limit=1)

def test_reject_invalid_copied_evidence(env,monkeypatch):
    g,cfg,conn=env
    output=extraction(); output['trait_ids']=['trait-invented']
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: SimpleNamespace(output=output))
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['failures']
    assert not storage.records(conn,'character_moments')

def test_expansion_is_bounded_and_keeps_parent_trait(env,monkeypatch):
    g,cfg,conn=env
    g.load_constitution(cfg).seeds=({'id':'seed-a','trait_id':'trait-a','prompt':'Explain a task'},)
    cfg.data['character']['max_seed_expansions']=1
    arms=[]
    def fake(stage,prompt,schema,**kw):
        import json
        data=json.loads(prompt)
        if stage=='character_generate' and 'seed' in data:
            return SimpleNamespace(output={'prompt':'A different task'})
        arm='chosen' if 'Approved constitution:' in kw['system_prompt'] else 'baseline'
        arms.append(arm)
        return SimpleNamespace(output={'response':arm})
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1,expand_seeds=True)['generated']==2
    seeds=storage.records(conn,'character_seeds')
    assert len(seeds)==2 and any(s.get('parent_expansion_id') for s in seeds)

def test_readonly_snapshot_and_pair_failure_keeps_extraction(env,monkeypatch):
    g,cfg,conn=env
    readonly=sqlite3.connect('file:'+str(cfg.db_path)+'?mode=ro',uri=True)
    def fake(stage,*a,**kw):
        if stage=='character_extract': return SimpleNamespace(output=extraction())
        raise RuntimeError('private provider error')
    monkeypatch.setattr(g.backend,'generate',fake)
    report=g.generate_pairs(readonly,cfg,source='corrections',limit=1)
    assert report['failures'] and 'private' not in str(report)
    assert len(storage.records(conn,'character_moments'))==1
    assert not storage.records(conn,'character_pairs')
    assert g.estimate(readonly,cfg,source='corrections',limit=1)['source_items']==1
    readonly.close()

def test_source_race_blocks_extraction(env,monkeypatch):
    g,cfg,conn=env
    def fake(*a,**k):
        conn.execute("UPDATE sessions SET source_revision='r2'")
        conn.commit()
        return SimpleNamespace(output=extraction())
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['failures']
    assert not storage.records(conn,'character_moments')

def test_synthetic_failure_does_not_persist_seed(env,monkeypatch):
    g,cfg,conn=env
    g.load_constitution(cfg).seeds=({'id':'seed-a','trait_id':'trait-a','prompt':'Task'},)
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('failed')))
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1)['failures']
    assert storage.records(conn,'character_seeds')==[]

def test_positive_correction_requires_traits_and_rationale(env,monkeypatch):
    g,cfg,conn=env
    output=extraction(); output['trait_ids']=[]; output['rationale']=''
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: SimpleNamespace(output=output))
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['failures']
    assert not storage.records(conn,'character_moments')

def test_mixed_seedless_traits_and_template_fingerprint(env,monkeypatch,tmp_path):
    g,cfg,conn=env
    doc=g.load_constitution(cfg)
    doc.traits['trait-b']='Be clear. Why: understanding'
    doc.seeds=({'id':'seed-a','trait_id':'trait-a','prompt':'Task'},)
    report=g.estimate(conn,cfg,source='synthetic',limit=1)
    assert report['skipped']['seedless']==1
    assert report['skipped']['seedless_traits']==['trait-b']
    templates=tmp_path/'templates'; templates.mkdir()
    for path in g.PROMPTS.glob('character-*-v1.md'):
        (templates/path.name).write_text(path.read_text())
    monkeypatch.setattr(g,'PROMPTS',templates)
    first=g.prompt_version()
    with (templates/'character-response-v1.md').open('a') as handle: handle.write(' Changed instruction.')
    assert g.prompt_version()!=first

def test_identical_arm_payloads_and_call_provenance(env,monkeypatch):
    g,cfg,conn=env
    doc=g.load_constitution(cfg)
    doc.seeds=({'id':'seed-a','trait_id':'trait-a','prompt':'Task'},)
    prompts=[]
    def fake(stage,prompt,schema,**kw):
        prompts.append(prompt)
        return SimpleNamespace(output={'response':'chosen' if len(prompts)%2 else 'baseline'},model='reported')
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1)['generated']==1
    assert prompts[0]==prompts[1]
    pair=storage.records(conn,'character_pairs')[0]
    call=pair['generator_calls'][0]
    assert call['prompt_sha256'] and call['started_at']<=call['finished_at']
    assert call['reported_model']=='reported' and 'configured_model' in call
    first=pair['seed_id']
    doc.sha256='new-constitution'
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1)['generated']==1
    assert len({p['seed_id'] for p in storage.records(conn,'character_pairs')})==2

@pytest.mark.parametrize('marker', ['line\nbreak', 'a"quote', 'a\\path'])
def test_raw_retention_markers(env,marker):
    g,cfg,conn=env
    from retention_policy import RetentionPolicy
    assert not g._allowed({'nested':[marker]},RetentionPolicy(text_markers=(marker,)))

def test_empty_and_cross_message_excerpt_rejected(env,monkeypatch):
    g,cfg,conn=env
    output=extraction(); output['context_excerpt']=''
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: SimpleNamespace(output=output))
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['failures']
    assert not storage.records(conn,'character_moments')

def test_failed_expansion_and_child_retry_without_refresh(env,monkeypatch):
    g,cfg,conn=env
    g.load_constitution(cfg).seeds=({'id':'seed-a','trait_id':'trait-a','prompt':'Task'},)
    cfg.data['character']['max_seed_expansions']=1
    state={'fail_expansion':True,'fail_child':True,'expansions':0}
    def fake(stage,prompt,schema,**kw):
        import json
        data=json.loads(prompt)
        if 'seed' in data:
            state['expansions']+=1
            if state['fail_expansion']: raise RuntimeError('failed expansion')
            return SimpleNamespace(output={'prompt':'Child task'})
        if 'Child task' in prompt and state['fail_child']: raise RuntimeError('failed child')
        return SimpleNamespace(output={'response':'chosen' if 'Approved constitution:' in kw['system_prompt'] else 'baseline'})
    monkeypatch.setattr(g.backend,'generate',fake)
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1,expand_seeds=True)['generated']==1
    state['fail_expansion']=False
    assert g.estimate(conn,cfg,source='synthetic',limit=1,expand_seeds=True)['source_items']==1
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1,expand_seeds=True)['failures']
    state['fail_child']=False
    assert g.generate_pairs(conn,cfg,source='synthetic',limit=1)['generated']==1
    assert state['expansions']==2

def test_excerpt_cannot_cross_original_message_boundary(env,monkeypatch):
    g,cfg,conn=env
    conn.execute('DELETE FROM messages')
    conn.executemany('INSERT INTO messages(session_id,seq,role,text) VALUES (?,?,?,?)', [('s',1,'user','first'),('s',2,'assistant','middle'),('s',3,'user','last'),('s',4,'assistant','Long answer'),('s',5,'user','Nie tak, shorter please')])
    conn.commit()
    output=extraction(); output['context_excerpt']='first\nmiddle'
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: SimpleNamespace(output=output))
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['failures']
    assert not storage.records(conn,'character_moments')

def test_identity_marker_excluded_before_calls(env,monkeypatch):
    g,cfg,conn=env
    cfg.memory_root.mkdir()
    (cfg.memory_root/'.dream-retention.json').write_text('{"text_markers":["private"]}')
    conn.execute("UPDATE sessions SET jsonl_path='/private/source'"); conn.commit()
    monkeypatch.setattr(g.backend,'generate',lambda *a,**k: pytest.fail('unexpected call'))
    assert g.generate_pairs(conn,cfg,source='corrections',limit=1)['source_items']==0

def test_refresh_estimate_has_attempt_and_supersedes(env,monkeypatch):
    g,cfg,conn=env
    monkeypatch.setattr(g.backend,'generate',lambda stage,*a,**k: SimpleNamespace(output=extraction() if stage=='character_extract' else {'response':'Brief answer'}))
    g.generate_pairs(conn,cfg,source='corrections',limit=1)
    pair=storage.records(conn,'character_pairs')[0]
    work=g.estimate(conn,cfg,source='corrections',limit=1,refresh=True)['work_items'][0]
    assert work['generation_attempt']==2 and work['supersedes_id']==pair['id']
