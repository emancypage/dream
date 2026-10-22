"""Bounded, policy-checked preference generation from visible source snapshots."""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import json
import logging
import sqlite3
import uuid
import backend
from config import provider_family
from model_types import validate_output
from character.constitution import load_constitution
from character import storage
from character.safety import policy_for, check_constitution, policy_fingerprint
from character.types import content_hash, canonical_json, RESPONSE_SCHEMA

class GenerationValidationError(ValueError):
    """Host-authored validation reason, safe to include in stage reports."""


PROMPTS = Path(__file__).resolve().parent.parent / 'prompts'
def prompt_version():
    return content_hash({'templates':{name:(PROMPTS/name).read_text() for name in ('character-extract-v1.md','character-expand-v1.md','character-response-v1.md')},'schemas':[EXTRACT_SCHEMA,EXPAND_SCHEMA,RESPONSE_SCHEMA]})
EXTRACT_SCHEMA = {'type':'object','properties':{
    **{k:{'type':'boolean'} for k in ('is_correction','context_sufficient')},
    **{k:{'type':'string'} for k in ('context_excerpt','corrected_assistant_response','user_correction','rationale')},
    'trait_ids':{'type':'array','items':{'type':'string'}},
    'agreement':{'type':'string','enum':['agrees','conflict','unclear']}},
    'additionalProperties':False}
EXTRACT_SCHEMA['required'] = list(EXTRACT_SCHEMA['properties'])
EXPAND_SCHEMA = {'type':'object','properties':{'prompt':{'type':'string'}},'required':['prompt'],'additionalProperties':False}
CUE = re.compile(r"(?<!\w)(?:nie(?: tak)?|źle|cofnij|stop|don't|actually|that's wrong)(?!\w)", re.I)


def public_probe_inputs():
    data = json.loads((PROMPTS.parent / 'fixtures/character/public.json').read_text(encoding='utf-8'))
    return ({probe['id'] for probe in data['probes']},
            {message['content'].strip() for probe in data['probes'] for message in probe['messages']})


def is_public_input(messages, seed_id=None):
    ids, texts = public_probe_inputs()
    return seed_id in ids or any(message['content'].strip() in texts for message in messages)


def _allowed(value, policy):
    return storage._text_allowed(value, policy)


def _source(conn, session_id):
    row=conn.execute('SELECT session_id,source,source_revision,parser_version,project_slug,jsonl_path,jsonl_mtime FROM sessions WHERE session_id=?',(session_id,)).fetchone()
    if row is None:
        return None
    result=dict(zip(('session_id','source','source_revision','parser_version','project_slug','source_path','jsonl_mtime'),row))
    result['messages']=[{'seq':r[0],'role':r[1],'content':r[2]} for r in conn.execute("SELECT seq,role,text FROM messages WHERE session_id=? AND role IN ('user','assistant') ORDER BY seq",(session_id,))]
    return result


def _select(conn, config, source, limit, refresh, expand_seeds=False):
    if source not in ('corrections','synthetic','all') or type(limit) is not int or limit <= 0:
        raise ValueError('source and positive limit required')
    doc=load_constitution(config); policy=policy_for(config); check_constitution(doc,policy)
    skipped={'oversized':0,'missing_context':0,'excluded':0,'completed':0,'seedless':0}
    items=[]; settings=config.data['character']
    moments=storage.eligible_records(conn,config,'character_moments',doc); pairs=storage.eligible_records(conn,config,'character_pairs',doc)
    def add(item):
        if is_public_input(item['messages'], item.get('seed', {}).get('id')):
            skipped['public_probe']=skipped.get('public_probe',0)+1
            return
        item['work_key']=item.pop('_work_key',None) or content_hash({'input':item,'constitution':doc.sha256,'prompt_version':prompt_version(),'policy':policy_fingerprint(policy),'bounds':{'context':settings['max_context_codepoints'],'response':settings['max_response_codepoints']},'extract':config.stage('character_extract'),'generate':config.stage('character_generate')})
        previous=[p for p in pairs if p.get('work_key')==item['work_key']]
        prior_moments=[m for m in moments if m.get('work_key')==item['work_key']]
        item['attempt']=max([p.get('generation_attempt',1) for p in previous]+[m['generation_attempt'] for m in prior_moments]+[0])+1
        item['supersedes_id']=max(previous,key=lambda p:p.get('generation_attempt',1))['id'] if previous else None
        old=max(prior_moments,key=lambda m:m['generation_attempt']) if prior_moments else None
        expansion_only=False
        if previous and item['kind']=='synthetic' and expand_seeds and settings['max_seed_expansions']>0:
            expanded=next((r for r in storage.eligible_records(conn,config,'character_seeds',doc) if r.get('expansion_work_key')==item['work_key']),None)
            child_pairs=[p for p in pairs if expanded and p.get('seed_id')==expanded['id']]
            expansion_only=not refresh and expanded is None
        item['expansion_only']=expansion_only
        if not refresh and (previous and not expansion_only or old and (not old.get('is_correction') or not old.get('context_sufficient'))):
            skipped['completed']+=1; return
        item['moment']=old if not refresh else None
        items.append(item)
    if source in ('corrections','all'):
        for (sid,) in conn.execute('SELECT session_id FROM sessions ORDER BY session_id'):
            identity=_source(conn,sid)
            if policy.excludes(project_slug=identity['project_slug'],source_path=identity['source_path']):
                skipped['excluded']+=1; continue
            messages=identity.pop('messages')
            for i,message in enumerate(messages):
                # Transcript user rows also contain tool notifications and loaded
                # skills. Their instructions are not human correction evidence.
                if message['role']=='user' and message['content'].lstrip().startswith(('<task-notification>', 'Base directory for this skill:')):
                    skipped['generated_message']=skipped.get('generated_message',0)+1
                    continue
                if message['role']!='user' or not CUE.search(message['content']) or i<1 or messages[i-1]['role']!='assistant':
                    continue
                prior=messages[:i-1]; context=[]; count=0
                for turn in reversed(prior):
                    if count+len(turn['content'])>settings['max_context_codepoints']:
                        break
                    context.insert(0,turn); count+=len(turn['content'])
                if not context or context[0]['role']!='user':
                    skipped['oversized' if prior else 'missing_context']+=1; continue
                rejected=messages[i-1]['content']
                if len(rejected)>settings['max_response_codepoints'] or len(message['content'])>settings['max_context_codepoints']:
                    skipped['oversized']+=1; continue
                if not _allowed([identity,context,rejected,message],policy):
                    skipped['excluded']+=1; continue
                add({'kind':'correction','identity':identity,'correction_seq':message['seq'],'assistant_seq':messages[i-1]['seq'],'messages':context,'rejected':rejected,'correction':message['content'],'omitted_message_count':len(prior)-len(context),'source_hash':content_hash(_source(conn,sid))})
    if source in ('synthetic','all'):
        skipped['seedless_traits']=sorted(set(doc.traits)-{seed['trait_id'] for seed in doc.seeds})
        skipped['seedless']=len(skipped['seedless_traits'])
        for seed in sorted(doc.seeds,key=lambda s:(s['trait_id'],s['id'])):
            seed=dict(seed)
            if not _allowed(seed,policy): skipped['excluded']+=1; continue
            if len(seed['prompt'])>settings['max_context_codepoints']: skipped['oversized']+=1; continue
            add({'kind':'synthetic','seed':seed,'messages':[{'role':'user','content':seed['prompt']}]})
        for stored in storage.eligible_records(conn,config,'character_seeds',doc):
            if stored.get('origin')!='expanded' or not _allowed(stored,policy):
                continue
            seed=stored['seed']
            if len(seed['prompt'])>settings['max_context_codepoints']:
                skipped['oversized']+=1; continue
            add({'kind':'synthetic','seed':seed,'stored_seed':stored,'messages':stored['prompt_messages'],'_work_key':stored['work_key'],'origin':'expanded','parent_expansion_id':stored['parent_expansion_id'],'expansion_call':stored['expansion_call']})
    return doc,policy,items[:limit],skipped


def estimate(conn, config, *, source, limit, refresh=False, expand_seeds=False):
    if not config.data['character']['enabled']: return {'source_items':0,'skipped':{'disabled':1},'failures':[]}
    doc,policy,items,skipped=_select(conn,config,source,limit,refresh,expand_seeds)
    corrections=sum(i['kind']=='correction' for i in items); seeds=len(items)-corrections
    expansions=min(sum(i['kind']=='synthetic' and not i.get('origin') for i in items),config.data['character']['max_seed_expansions']) if expand_seeds else 0
    return {'source_items':len(items),'work_ids':[i['work_key'] for i in items],'work_items':[{'work_id':i['work_key'],'generation_attempt':i['attempt'],'supersedes_id':i['supersedes_id']} for i in items],'bounded_codepoints':sum(len(canonical_json(i)) for i in items),'call_bounds':{'extraction':corrections,'responses':corrections+2*seeds+2*expansions,'expansion':expansions,'judging':2*(len(items)+expansions)},'routes':{s:dict(config.stage(s)) for s in ('character_extract','character_generate')},'skipped':skipped,'failures':[]}


def generate_pairs(conn, config, *, source, limit, refresh=False, expand_seeds=False):
    if not config.data['character']['enabled']: return {'generated':0,'failures':[],'skipped':{'disabled':1}}
    doc,policy,items,skipped=_select(conn,config,source,limit,refresh,expand_seeds)
    fingerprint=policy_fingerprint(policy); version=prompt_version()
    result={'generated':0,'source_items':len(items),'failures':[],'skipped':skipped}
    expansion_budget=config.data['character']['max_seed_expansions'] if expand_seeds else 0
    def check(item, writer=None):
        current=policy_for(config)
        if policy_fingerprint(current)!=fingerprint or load_constitution(config).sha256!=doc.sha256 or prompt_version()!=version:
            raise GenerationValidationError('inputs changed')
        check_constitution(doc,current)
        if item['kind']=='correction' and content_hash(_source(writer or conn,item['identity']['session_id']))!=item['source_hash']:
            raise GenerationValidationError('source changed')
    def call(item,stage,template,payload,schema,constitution=True):
        check(item)
        system=(PROMPTS/template).read_text()+policy.context()
        if constitution: system+='\nApproved constitution:\n'+doc.text
        started_at=datetime.now(timezone.utc).isoformat()
        prompt=canonical_json(payload)
        request_schema=schema
        if stage=='character_extract':
            # Constrain verbatim evidence at the provider boundary. Keep the
            # host's independent evidence checks below for every returned result.
            properties=dict(schema['properties'])
            properties['corrected_assistant_response']={'type':'string','enum':[item['rejected']]}
            properties['user_correction']={'type':'string','enum':[item['correction']]}
            request_schema=dict(schema,properties=properties)
        answer=backend.generate(stage,prompt,request_schema,config=config,system_prompt=system)
        finished_at=datetime.now(timezone.utc).isoformat()
        check(item); validate_output(answer.output,schema)
        if not _allowed(answer.output,policy): raise GenerationValidationError('excluded output')
        metadata={'prompt_sha256':content_hash({'system':system,'prompt':prompt,'schema':request_schema}),'configured_model':config.stage(stage).get('model'),'reported_model':getattr(answer,'model',None),'started_at':started_at,'finished_at':finished_at,'stage':stage,'family':provider_family(config,stage),'provider':getattr(answer,'provider',config.stage(stage)['provider']),'model':getattr(answer,'model',config.stage(stage).get('model')),'options':dict(config.stage(stage)),'duration_ms':getattr(answer,'duration_ms',None),'usage':asdict(answer.usage) if is_dataclass(getattr(answer,'usage',None)) else None}
        return answer.output,metadata
    def response(item,messages,arm):
        output,meta=call(item,'character_generate','character-response-v1.md',{'messages':messages},RESPONSE_SCHEMA,arm=='chosen')
        text=output['response']
        if not text.strip() or len(text)>config.data['character']['max_response_codepoints']: raise GenerationValidationError('invalid response length')
        return text,meta
    def persist(item,table,record,seed_record=None):
        check(item)
        with storage.open_writer(config) as writer:
            writer.execute('BEGIN IMMEDIATE')
            check(item,writer)
            if not _allowed(record,policy): raise GenerationValidationError('excluded record')
            if table=='character_pairs':
                if seed_record is not None:
                    stored=storage.get_record(writer,'character_seeds',seed_record['id'])
                    if stored is None:
                        stored=storage.insert_record(writer,'character_seeds',seed_record)
                    record['provenance']['source_content_sha256']=stored['content_sha256']
                existing=[r for r in storage.records(writer,table) if r.get('work_key')==record['work_key']]
                if any(r.get('generation_attempt',1)>=record['generation_attempt'] for r in existing):
                    raise GenerationValidationError('generation attempt changed')
                dependency=record.get('moment_id') or record.get('seed_id')
                parent_table='character_moments' if record.get('moment_id') else 'character_seeds'
                parent=storage.get_record(writer,parent_table,dependency)
                if parent is None or parent['content_sha256']!=record['provenance']['source_content_sha256']:
                    raise GenerationValidationError('source record changed')
            stored_record=storage.insert_record(writer,table,record)
            storage.mark_stale(writer,doc)
            return stored_record
    def schedule_expansion(item, parent_id):
        nonlocal expansion_budget
        if item['kind']!='synthetic' or item.get('origin') or expansion_budget<=0:
            return
        expansion_budget-=1
        stored=next((r for r in storage.eligible_records(conn,config,'character_seeds',doc) if r.get('expansion_work_key')==item['work_key']),None)
        if stored is None:
            expanded,expansion_call=call(item,'character_generate','character-expand-v1.md',{'seed':item['seed']},EXPAND_SCHEMA)
            prompt=expanded['prompt']
            if not prompt.strip() or len(prompt)>config.data['character']['max_context_codepoints'] or is_public_input([{'content':prompt}]):
                raise GenerationValidationError('invalid expansion length')
            child_seed={'id':item['seed']['id']+'-expanded-'+content_hash(prompt)[:16],'trait_id':item['seed']['trait_id'],'prompt':prompt}
            key=content_hash({'parent':item['work_key'],'prompt':prompt})
            record={'id':'seed-'+content_hash({'seed':child_seed,'constitution':doc.sha256}), 'work_key':key,'constitution_sha256':doc.sha256,'constitution_version':doc.version,'revision':1,'excluded':False,'origin':'expanded','expansion_work_key':item['work_key'],'parent_expansion_id':parent_id,'expansion_call':expansion_call,'human_seed_id':item['seed']['id'],'trait_id':item['seed']['trait_id'],'seed':child_seed,'prompt_messages':[{'role':'user','content':prompt}],'provenance':{'parent_seed_id':parent_id}}
            stored=persist(item,'character_seeds',record)
        child_seed=stored['seed']; key=stored['work_key']
        previous=[p for p in storage.records(conn,'character_pairs') if p.get('work_key')==key]
        if refresh or not previous:
            items.append({'kind':'synthetic','seed':child_seed,'stored_seed':stored,'messages':stored['prompt_messages'],'work_key':key,'attempt':max((p['generation_attempt'] for p in previous),default=0)+1,'supersedes_id':max(previous,key=lambda p:p['generation_attempt'])['id'] if previous else None,'origin':'expanded','parent_expansion_id':stored['parent_expansion_id'],'expansion_call':stored['expansion_call']})
    for item in items:
        try:
            if item.get('expansion_only'):
                parent_id='seed-'+content_hash({'seed':item['seed'],'constitution':doc.sha256})
                schedule_expansion(item,parent_id)
                continue
            moment=item.get('moment'); calls=[]; seed_record=None
            base={'work_key':item['work_key'],'constitution_sha256':doc.sha256,'constitution_version':doc.version,'prompt_version':prompt_version(),'created_at':datetime.now(timezone.utc).isoformat(),'revision':1,'excluded':False}
            provenance={'source_kind':item['kind'],'constitution_sha256':doc.sha256,'reconstruction_hash':content_hash(item['messages']),'context_status':'bounded_visible_reconstruction' if item['kind']=='correction' else 'approved_human_seed','omitted_message_count':item.get('omitted_message_count',0)}
            if item['kind']=='correction':
                provenance.update(item['identity']); provenance['historical_family']='anthropic' if item['identity']['source']=='claude' else 'unknown'
                provenance['message_range']=[item['messages'][0]['seq'],item['messages'][-1]['seq']]
                if moment is None:
                    output,meta=call(item,'character_extract','character-extract-v1.md',{'messages':item['messages'],'assistant_response':item['rejected'],'user_correction':item['correction'],'trait_ids':list(doc.traits)},EXTRACT_SCHEMA)
                    if output['is_correction'] and output['context_sufficient'] and (not output['trait_ids'] or len(set(output['trait_ids']))!=len(output['trait_ids']) or not output['rationale'].strip()):
                        raise GenerationValidationError('positive correction needs traits and rationale')
                    if not output['context_excerpt'].strip() or not any(output['context_excerpt'] in m['content'] for m in item['messages']):
                        raise GenerationValidationError('context excerpt is not copied')
                    if output['corrected_assistant_response']!=item['rejected']:
                        raise GenerationValidationError('assistant response is not copied')
                    if output['user_correction']!=item['correction']:
                        raise GenerationValidationError('user correction is not copied')
                    if any(t not in doc.traits for t in output['trait_ids']):
                        raise GenerationValidationError('unknown extraction trait')
                    if len(output['rationale'])>2000:
                        raise GenerationValidationError('extraction rationale is too long')
                    moment=persist(item,'character_moments',dict(base,**output,generation_attempt=item['attempt'],id='moment-'+uuid.uuid4().hex,provenance=provenance,**item['identity'],assistant_seq=item['assistant_seq'],correction_seq=item['correction_seq'],prompt_messages=item['messages'],extraction=meta,disposition='completed'))
                if not moment['is_correction'] or not moment['context_sufficient']:
                    skipped['negative']=skipped.get('negative',0)+1; continue
                chosen,meta=response(item,item['messages'],'chosen'); calls=[meta]; rejected=item['rejected']; traits=moment['trait_ids']; rationale=moment['rationale']; conflict=moment['agreement']=='conflict'
                provenance['source_content_sha256']=moment['content_sha256']
                source_fields={'moment_id':moment['id'],**item['identity']}
            else:
                seed=item['seed']; messages=item['messages']
                seed_record=dict(base,id='seed-'+content_hash({'seed':seed,'constitution':doc.sha256}),human_seed_id=seed['id'],human_seed_hash=content_hash(seed),trait_id=seed['trait_id'],prompt_messages=messages,origin=item.get('origin','human'),parent_expansion_id=item.get('parent_expansion_id'),expansion_call=item.get('expansion_call'),provenance={'source_kind':'human_seed','human_seed_id':seed['id']})
                stored=item.get('stored_seed') or seed_record
                if item.get('stored_seed'): seed_record=None
                chosen,a=response(item,messages,'chosen'); rejected,b=response(item,messages,'baseline'); calls=[a,b]
                provenance['source_content_sha256']=stored.get('content_sha256')
                provenance['human_seed_id']=stored['human_seed_id']
                if stored.get('parent_expansion_id'):
                    provenance['parent_expansion_id']=stored['parent_expansion_id']
                traits=[seed['trait_id']]; rationale='Synthetic contrast for approved trait'; conflict=False; source_fields={'seed_id':stored['id']}
            if chosen.strip()==rejected.strip(): raise GenerationValidationError('equal responses')
            pair=dict(base,id='pair-'+uuid.uuid4().hex,generation_attempt=item['attempt'],supersedes_id=item['supersedes_id'],source_kind=item['kind'],prompt_messages=item['messages'],chosen=chosen,rejected=rejected,rationale=rationale,trait_ids=traits,conflict=conflict,provenance=provenance,generator_calls=calls,status='pending',edited=False,**source_fields)
            persist(item,'character_pairs',pair,seed_record=seed_record); result['generated']+=1
            schedule_expansion(item,source_fields.get('seed_id'))
        except sqlite3.Error:
            raise
        except (ValueError,RuntimeError,TimeoutError,OSError) as exc:
            reason=str(exc) if isinstance(exc,GenerationValidationError) else 'model or runtime failure'
            error=type(exc).__name__
            logging.getLogger(__name__).warning('Character generation failed for %s: %s: %s',item['work_key'],error,reason)
            result['failures'].append({'work_id':item['work_key'],'error':error,'reason':reason})
    return result
