"""Independent, advisory, swapped-order character judgments."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
import sqlite3
import uuid
import backend
from config import provider_family
from model_types import validate_output
from character import storage
from character.constitution import load_constitution
from character.safety import policy_for,check_constitution,policy_fingerprint
from character.types import content_hash,canonical_json
from character.review import candidate_hash

PROMPT=Path(__file__).resolve().parent.parent/'prompts'/'character-judge-v1.md'
ASSESSMENT={'type':'object','properties':{'trait_id':{'type':'string'},'a_assessment':{'type':'string','enum':['supports','violates','unclear']},'b_assessment':{'type':'string','enum':['supports','violates','unclear']},'rationale':{'type':'string'}},'required':['trait_id','a_assessment','b_assessment','rationale'],'additionalProperties':False}
JUDGE_SCHEMA={'type':'object','properties':{'winner':{'type':'string','enum':['A','B','tie','neither']},'assessments':{'type':'array','items':ASSESSMENT},'rationale':{'type':'string'}},'required':['winner','assessments','rationale'],'additionalProperties':False}


def prompt_version():
    return content_hash({'template':PROMPT.read_text(),'schema':JUDGE_SCHEMA})


def _normalize(output,swapped,traits):
    validate_output(output,JUDGE_SCHEMA)
    assessments=output['assessments']
    if {a['trait_id'] for a in assessments}!=set(traits) or len(assessments)!=len(traits) or not output['rationale'].strip() or len(output['rationale'])>2000 or any(not a['rationale'].strip() or len(a['rationale'])>2000 for a in assessments): raise ValueError('invalid judge trait coverage or rationale')
    winner=output['winner']
    normalized={'A':'rejected' if swapped else 'chosen','B':'chosen' if swapped else 'rejected','tie':'tie','neither':'neither'}[winner]
    labels={a['trait_id']:{'chosen':a['b_assessment'] if swapped else a['a_assessment'],'rejected':a['a_assessment'] if swapped else a['b_assessment']} for a in assessments}
    return {'winner':normalized,'assessments':labels}


def judge_pairs(conn,config,*,limit):
    if type(limit) is not int or limit<=0: raise ValueError('positive limit required')
    result={'judged':0,'failures':[],'skipped':{}}
    if not config.data['character']['enabled']: result['skipped']['disabled']=1; return result
    doc=load_constitution(config); policy=policy_for(config); check_constitution(doc,policy)
    fingerprint=policy_fingerprint(policy); version=prompt_version()
    family=provider_family(config,'character_judge')
    other_families={provider_family(config,s) for s in ('character_extract','character_generate')}
    override=config.data['character']['allow_same_family_judge']
    if family in other_families and not override: raise ValueError('judge must use an independent family')
    identity=content_hash({'route':config.stage('character_judge'),'family':family,'prompt_version':version})
    done=storage.eligible_records(conn,config,'character_judgments',doc)
    pairs=[p for p in storage.eligible_records(conn,config,'character_pairs',doc) if p.get('status') not in ('accepted','rejected','stale') and not any(j.get('pair_id')==p['id'] and j.get('pair_revision')==p['revision'] and j.get('pair_sha256')==p['content_sha256'] and j.get('judge_identity')==identity and j.get('complete') is True for j in done)][:limit]
    def check(pair,db):
        current=policy_for(config)
        if load_constitution(config).sha256!=doc.sha256 or policy_fingerprint(current)!=fingerprint or prompt_version()!=version: raise ValueError('inputs changed')
        check_constitution(doc,current)
        live=next((p for p in storage.eligible_records(db,config,'character_pairs',doc) if p['id']==pair['id']),None)
        if live is None or live['content_sha256']!=pair['content_sha256']: raise ValueError('pair changed or unavailable')
    for pair in pairs:
        try:
            check(pair,conn)
            if not pair.get('trait_ids') or any(t not in doc.traits for t in pair['trait_ids']): raise ValueError('invalid traits')
            evidence=None
            if pair.get('moment_id'):
                evidence=next((m for m in storage.eligible_records(conn,config,'character_moments',doc) if m['id']==pair['moment_id']),None)
                if evidence is None: raise ValueError('source unavailable')
                evidence={'user_correction':evidence.get('user_correction'),'rationale':evidence.get('rationale')}
            outputs=[]; calls=[]; normalized=[]
            historical_family=pair.get('provenance',{}).get('historical_family','anthropic' if pair.get('source')=='claude' or pair.get('provenance',{}).get('source')=='claude' else 'unknown')
            overlap=historical_family==family
            for swapped in (False,True):
                check(pair,conn)
                system=PROMPT.read_text()+policy.context()+'\nApproved constitution:\n'+doc.text
                payload={'messages':pair['prompt_messages'],'traits':{t:doc.traits[t] for t in pair['trait_ids']},'A':pair['rejected'] if swapped else pair['chosen'],'B':pair['chosen'] if swapped else pair['rejected'],'correction_evidence':evidence}
                prompt=canonical_json(payload); start=datetime.now(timezone.utc).isoformat()
                answer=backend.generate('character_judge',prompt,JUDGE_SCHEMA,config=config,system_prompt=system)
                end=datetime.now(timezone.utc).isoformat(); check(pair,conn)
                output=answer.output
                normalized.append(_normalize(output,swapped,pair['trait_ids']))
                if not storage._text_allowed(output,policy): raise ValueError('excluded judgment')
                outputs.append(output)
                calls.append({'stage':'character_judge','provider':getattr(answer,'provider',config.stage('character_judge')['provider']),'family':family,'configured_model':config.stage('character_judge').get('model'),'reported_model':getattr(answer,'model',None),'options':dict(config.stage('character_judge')),'started_at':start,'finished_at':end,'duration_ms':getattr(answer,'duration_ms',None),'usage':asdict(answer.usage) if is_dataclass(getattr(answer,'usage',None)) else None,'prompt_sha256':content_hash({'system':system,'prompt':prompt,'schema':JUDGE_SCHEMA}),'swapped':swapped,'same_family_override':override,'historical_overlap_warning':overlap})
            verdict=normalized[0]['winner'] if normalized[0]==normalized[1] else 'inconsistent'
            record={'id':'judgment-'+uuid.uuid4().hex,'revision':1,'pair_id':pair['id'],'pair_revision':pair['revision'],'pair_sha256':pair['content_sha256'],'candidate_sha256':candidate_hash(pair),'constitution_sha256':doc.sha256,'constitution_version':doc.version,'complete':True,'created_at':datetime.now(timezone.utc).isoformat(),'verdicts':outputs,'normalized_verdicts':normalized,'normalized_result':verdict,'calls':calls,'judge_identity':identity,'prompt_version':version,'same_family_override':override,'historical_overlap_warning':overlap,'provenance':pair.get('provenance',{})}
            with storage.open_writer(config) as writer:
                writer.execute('BEGIN IMMEDIATE'); check(pair,writer)
                if any(j.get('pair_id')==pair['id'] and j.get('pair_revision')==pair['revision'] and j.get('judge_identity')==identity for j in storage.records(writer,'character_judgments')): raise ValueError('judgment already exists')
                storage.insert_record(writer,'character_judgments',record)
                storage.mark_stale(writer,doc)
            result['judged']+=1
        except sqlite3.Error: raise
        except (ValueError,RuntimeError,TimeoutError,OSError) as exc:
            logging.getLogger(__name__).warning('Character judgment failed for %s: %s',pair['id'],type(exc).__name__)
            result['failures'].append({'pair_id':pair['id'],'error':type(exc).__name__})
    return result
