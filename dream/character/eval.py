"""Fixed, paired, multi-turn evaluation with private policy-filtered reports."""
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import sqlite3
import uuid
import backend
from storage import open_db_readonly
from config import provider_family
from model_types import validate_output
from character import storage
from character.constitution import load_constitution
from character.render import render_block
from character.safety import policy_for, policy_fingerprint, check_constitution
from character.targets import make_target
from character.types import content_hash, canonical_json, RESPONSE_SCHEMA

PROMPTS = Path(__file__).resolve().parent.parent / 'prompts'
RESPONSE_PROMPT = PROMPTS / 'character-eval-v1.md'
JUDGE_PROMPT = PROMPTS / 'character-eval-judge-v1.md'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fixtures(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {'version', 'probes'} or not isinstance(value['version'], str) or not value['version'] or not isinstance(value['probes'], list):
        raise ValueError('invalid fixture document')
    seen = set()
    for p in value['probes']:
        if not isinstance(p, dict) or set(p) != {'id','category','messages','trait_ids','trait_sha256','rubric'}:
            raise ValueError('invalid probe fields')
        if any(not isinstance(p[k], str) or not p[k].strip() or len(p[k]) > 100 for k in ('id','category')) or p['id'] in seen:
            raise ValueError('invalid probe identity')
        seen.add(p['id'])
        if not isinstance(p['messages'], list) or not p['messages'] or any(not isinstance(m,dict) or set(m) != {'role','content'} or m['role'] != 'user' or not isinstance(m['content'],str) or not m['content'].strip() for m in p['messages']):
            raise ValueError('probes require scripted user turns')
        if not isinstance(p['trait_ids'],list) or any(not isinstance(t,str) for t in p['trait_ids']) or len(set(p['trait_ids'])) != len(p['trait_ids']) or not isinstance(p['trait_sha256'],dict) or set(p['trait_sha256']) != set(p['trait_ids']) or any(not isinstance(h,str) for h in p['trait_sha256'].values()):
            raise ValueError('invalid trait bindings')
        if not isinstance(p['rubric'],dict) or not p['rubric'] or any(not isinstance(d,str) or not d.strip() or not isinstance(anchors,dict) or set(anchors) != set('01234') or any(not isinstance(a,str) or not a.strip() for a in anchors.values()) for d,anchors in p['rubric'].items()):
            raise ValueError('invalid anchored rubric')
    return value


def _schema(probe):
    dimensions = {d:{'type':'integer','enum':list(range(5))} for d in probe['rubric']}
    scores = {'type':'object','properties':dimensions,'required':list(dimensions),'additionalProperties':False}
    return {'type':'object','properties':{'scores':{'type':'object','properties':{'A':scores,'B':scores},'required':['A','B'],'additionalProperties':False},'rationale':{'type':'string'}},'required':['scores','rationale'],'additionalProperties':False}


def _scores(value, schema, swapped):
    validate_output(value,schema)
    reason = value['rationale']
    if not reason.strip() or len(reason) > 2000 or not any('a' <= c.lower() <= 'z' for c in reason):
        raise ValueError('invalid English rationale')
    return {'baseline':value['scores']['B' if swapped else 'A'], 'A':value['scores']['A' if swapped else 'B']}


def _call(config, stage, prompt, system, schema, calls, invoke=None, **extra):
    route = dict(config.stage(stage))
    family = provider_family(config,stage)
    metadata = {'stage':stage,'provider':route['provider'],'auth_family':family,'family':family,
                'configured_model':route.get('model'),'reported_model':None,'options':route,
                'started_at':_now(),'prompt_sha256':content_hash({'system':system,'prompt':prompt,'schema':schema}),**extra}
    try:
        answer = invoke() if invoke else backend.generate(stage,prompt,schema,config=config,system_prompt=system)
        metadata.update(provider=getattr(answer,'provider',route['provider']),reported_model=getattr(answer,'model',None),
                        duration_ms=getattr(answer,'duration_ms',None),usage=asdict(answer.usage) if is_dataclass(getattr(answer,'usage',None)) else None)
        return answer.output
    except sqlite3.Error:
        raise
    except Exception:
        metadata['failure_code']='provider_failed'
        raise
    finally:
        metadata['finished_at']=_now()
        metadata.setdefault('duration_ms',int((datetime.fromisoformat(metadata['finished_at'])-datetime.fromisoformat(metadata['started_at'])).total_seconds()*1000))
        metadata.setdefault('usage',None)
        calls.append(metadata)


def evaluate(conn, config, *, fixtures, limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError('positive limit required')
    if not config.data['character']['enabled']:
        return {'run_id':None,'failures':[],'skipped':{'disabled':1}}
    document = load_constitution(config)
    policy = policy_for(config)
    check_constitution(document,policy)
    fixture = _fixtures(fixtures)
    rendered = render_block(conn,config)
    templates = {'response':RESPONSE_PROMPT.read_text(),'judge':JUDGE_PROMPT.read_text()}
    fingerprint = policy_fingerprint(policy)
    family = provider_family(config,'character_judge')
    override = config.data['character']['allow_same_family_judge']
    if family in {provider_family(config,s) for s in ('character_extract','character_generate','character_eval_response')} and not override:
        raise ValueError('evaluation judge must use an independent family')
    target = make_target(config)
    routes = {s:dict(config.stage(s)) for s in ('character_eval_response','character_judge')}
    for route in routes.values():
        if not route.get('model'):
            raise ValueError('evaluation requires explicitly configured models')
    allowed_probes=[p for p in fixture['probes'] if storage._text_allowed(p,policy)]
    policy_skipped=len(fixture['probes'])-len(allowed_probes)
    prompt_version=content_hash(templates)
    run_id = 'eval-' + uuid.uuid4().hex
    manifest = dict(rendered['manifest'],run_id=run_id)
    provider_snapshot=content_hash({s:config.provider(r['provider']) for s,r in routes.items()})
    identities = {'fixture_sha256':content_hash(fixture),'rubric_sha256':content_hash({p['id']:p['rubric'] for p in fixture['probes']}),
                  'template_sha256':content_hash(templates),'routes_sha256':content_hash({'routes':routes,'providers_sha256':provider_snapshot}),
                  'render_sha256':content_hash(rendered),'constitution_sha256':document.sha256}
    run = {'id':run_id,'run_id':run_id,'revision':1,'constitution_sha256':document.sha256,'constitution_version':document.version,
           'created_at':_now(),'manifest':manifest,'pair_ids':manifest['pair_ids'],'identities':identities,
           'response_route':routes['character_eval_response'],'judge_route':routes['character_judge'],
           'prompt_version':prompt_version,'policy_skipped':policy_skipped,
           'targets':{'baseline':target.identity(),'A':target.identity()},'fixture_version':fixture['version'],
           'same_family_override':override,'historical_overlap_warning':manifest.get('historical_overlap_warning',False),
           'probes':[{'id':p['id'],'category':p['category']} for p in allowed_probes[:limit]],'state':'running'}
    failures=[]

    def check(db):
        if db is conn:
            fresh=open_db_readonly(config.db_path)
            try:
                return check(fresh)
            finally:
                fresh.close()
        current=policy_for(config)
        if load_constitution(config).sha256 != document.sha256 or policy_fingerprint(current) != fingerprint or _fixtures(fixtures) != fixture or {'response':RESPONSE_PROMPT.read_text(),'judge':JUDGE_PROMPT.read_text()} != templates or any(dict(config.stage(s)) != route for s,route in routes.items()) or content_hash({s:config.provider(r['provider']) for s,r in routes.items()}) != provider_snapshot:
            raise ValueError('evaluation inputs changed')
        check_constitution(document,current)
        if render_block(db,config) != rendered or not storage._text_allowed(rendered,current):
            raise ValueError('rendered context changed or excluded')
        return current

    def save(record, existing=False):
        with storage.open_writer(config) as writer:
            writer.execute('BEGIN IMMEDIATE')
            current=check(writer)
            if not storage._text_allowed(record,current) or not storage._text_allowed(run,current):
                raise ValueError('evaluation output excluded')
            if storage.get_record(writer,'character_eval_runs',run_id) is None:
                storage.insert_record(writer,'character_eval_runs',run)
            if record.get('responses'):
                storage.mark_stale(writer,document)
            if existing:
                old=storage.get_record(writer,'character_eval_results',record['id'])
                record=dict(record,revision=old['revision']+1)
                storage.update_record(writer,'character_eval_results',record,expected_sha=old['content_sha256'])
            else:
                storage.insert_record(writer,'character_eval_results',record)

    for probe in allowed_probes[:limit]:
        base={'probe_input':probe,'run_id':run_id,'probe_id':probe['id'],'category':probe['category'],'constitution_sha256':document.sha256,'constitution_version':document.version,'prompt_version':prompt_version,'created_at':_now(),'revision':1}
        supported=all(t in document.traits and content_hash(document.traits[t]) == probe['trait_sha256'][t] for t in probe['trait_ids'])
        if not supported:
            reason='unsupported'
            for label in ('baseline','A'):
                save(dict(base,id='result-'+uuid.uuid4().hex,target=label,state=reason,calls=[]))
            continue
        rows={}
        for label in ('baseline','A'):
            row=dict(base,id='result-'+uuid.uuid4().hex,target=label,state='generated',calls=[],responses=[])
            try:
                messages=[]
                for message in probe['messages']:
                    current=check(conn)
                    messages.append(dict(message))
                    block=rendered['text'] if label=='A' else ''
                    settings={'instructions':templates['response'],'policy_context':current.context()}
                    system=templates['response']+ ('\n'+block if block else '')+current.context()
                    prompt=canonical_json({'messages':messages})
                    if len(system)+len(prompt) > config.data['character']['max_context_codepoints']:
                        raise ValueError('probe context exceeds configured bound')
                    output=_call(config,'character_eval_response',prompt,system,RESPONSE_SCHEMA,row['calls'],
                                 invoke=lambda:target.respond(messages,block,settings),target=label,same_family_override=override,historical_overlap_warning=manifest.get('historical_overlap_warning',False))
                    validate_output(output,RESPONSE_SCHEMA)
                    if not output['response'].strip() or len(output['response']) > config.data['character']['max_response_codepoints'] or not storage._text_allowed(output,check(conn)):
                        raise ValueError('invalid response')
                    row['responses'].append(output['response'])
                    messages.append({'role':'assistant','content':output['response']})
                row['trajectory']=messages
                save(row)
                rows[label]=row
            except sqlite3.Error:
                raise
            except (ValueError,RuntimeError,TimeoutError,OSError):
                failures.append({'probe_id':probe['id'],'target':label,'error':'response_failed'})
                # Never persist a partial trajectory as a validated response.
                row.update(state='generation_failed',failure_code='response_failed',responses=[])
                try:
                    save(row)
                except sqlite3.Error:
                    raise
                except (ValueError,RuntimeError,OSError):
                    pass  # A changed policy/input discards this item completely.
        if set(rows) != {'baseline','A'}:
            continue
        calls=[]; outputs=[]; normalized=[]
        try:
            for swapped in (False,True):
                current=check(conn)
                payload={'rubric':probe['rubric'],'constitution':document.text,
                         'A':rows['A' if swapped else 'baseline']['trajectory'],
                         'B':rows['baseline' if swapped else 'A']['trajectory']}
                prompt=canonical_json(payload); system=templates['judge']+current.context()
                schema=_schema(probe)
                output=_call(config,'character_judge',prompt,system,schema,calls,swapped=swapped,same_family_override=override,historical_overlap_warning=manifest.get('historical_overlap_warning',False))
                normalized.append(_scores(output,schema,swapped))
                if not storage._text_allowed(output,check(conn)):
                    raise ValueError('excluded score')
                outputs.append(output)
            state='scored' if normalized[0] == normalized[1] else 'inconsistent'
            # Persist both targets and both judgments atomically.
            with storage.open_writer(config) as writer:
                writer.execute('BEGIN IMMEDIATE'); current=check(writer)
                storage.mark_stale(writer,document)
                for label,row in rows.items():
                    record=dict(row,state=state,judgments=outputs,normalized_scores=normalized,calls=row['calls']+calls,
                                revision=2,same_family_override=override)
                    if state=='scored': record['scores']=normalized[0][label]
                    if not storage._text_allowed(record,current) or not storage._text_allowed(run,current): raise ValueError('excluded result')
                    old=storage.get_record(writer,'character_eval_results',row['id'])
                    storage.update_record(writer,'character_eval_results',record,expected_sha=old['content_sha256'])
        except sqlite3.Error:
            raise
        except (ValueError,RuntimeError,TimeoutError,OSError):
            failures.append({'probe_id':probe['id'],'error':'score_failed'})
            for row in rows.values():
                row.update(state='score_failed',failure_code='score_failed',calls=row['calls']+calls)
                try:
                    save(row,True)
                except sqlite3.Error:
                    raise
                except (ValueError,RuntimeError,OSError):
                    pass
    # A run with zero probes still has an auditable identity.
    try:
        with storage.open_writer(config) as writer:
            writer.execute('BEGIN IMMEDIATE'); check(writer)
            old=storage.get_record(writer,'character_eval_runs',run_id)
            final=dict(run,state='incomplete' if failures else 'complete',failures=failures,finished_at=_now())
            if not storage._text_allowed(final,policy_for(config)):
                raise ValueError('evaluation metadata excluded')
            if old:
                storage.update_record(writer,'character_eval_runs',dict(final,revision=old['revision']+1),expected_sha=old['content_sha256'])
            else:
                storage.insert_record(writer,'character_eval_runs',final)
    except sqlite3.Error:
        raise
    except (ValueError,RuntimeError,OSError):
        failures.append({'error':'inputs_changed'})
    return {'run_id':run_id,'failures':failures}


def report_run(conn, config, run_id):
    """Read only; never reconstruct scores from excluded or incomplete records."""
    document=load_constitution(config); check_constitution(document,policy_for(config))
    run=next((r for r in storage.eligible_records(conn,config,'character_eval_runs') if r['id']==run_id),None)
    if run is None:
        raise ValueError('evaluation run unavailable under current policy')
    rows=[r for r in storage.eligible_records(conn,config,'character_eval_results') if r.get('run_id')==run_id]
    categories=[]
    for category in sorted({p['category'] for p in run['probes']}):
        probes=[p for p in run['probes'] if p['category']==category]
        matching=[r for r in rows if r['category']==category]
        groups={p['id']:{r['target']:r for r in matching if r['probe_id']==p['id']} for p in probes}
        common=[g for g in groups.values() if set(g)=={'baseline','A'} and all(r['state']=='scored' for r in g.values())]
        def mean(label):
            values=[v for g in common for v in g[label]['scores'].values()]
            return sum(values)/len(values) if values else None
        dimensions={}
        for dimension in sorted({d for g in common for d in g['baseline']['scores']}):
            subset=[g for g in common if dimension in g['baseline']['scores'] and dimension in g['A']['scores']]
            before=sum(g['baseline']['scores'][dimension] for g in subset)/len(subset)
            after=sum(g['A']['scores'][dimension] for g in subset)/len(subset)
            dimensions[dimension]={'baseline':before,'A':after,'delta':after-before,'valid':len(subset)}
        baseline=mean('baseline'); treated=mean('A')
        counts={state:sum(any(r['state']==state for r in g.values()) for g in groups.values()) for state in ('unsupported','skipped','inconsistent')}
        failed=sum(any(r['state'] in ('generation_failed','score_failed','generated') for r in g.values()) for g in groups.values())
        categories.append(dict(category=category,dimensions=dimensions,total=len(probes),valid=len(common),baseline=baseline,A=treated,delta=treated-baseline if common else None,
                               failed=failed,omitted=len(probes)-len(common),**counts))
    manifest=dict(run['manifest'],run_id=run_id,identities=run['identities'],result_ids=[row['id'] for row in rows])
    lines=['| Category | Valid / total | Baseline | A | Delta | Inconsistent | Unsupported | Skipped | Failed |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(value): return 'n/a' if value is None else f'{value:.2f}'
    for row in categories:
        lines.append(f"| {' '.join(row['category'].replace('|',' ').split())} | {row['valid']} / {row['total']} | {number(row['baseline'])} | {number(row['A'])} | {number(row['delta'])} | {row['inconsistent']} | {row['unsupported']} | {row['skipped']} | {row['failed']} |")
    lines.append('\nModel judgments are evidence about these answers. Context evaluation does not establish weight-training robustness.')
    current_drift={'constitution':document.sha256 != run['constitution_sha256'],'policy':policy_fingerprint(policy_for(config)) != run['manifest']['policy_fingerprint']}
    return {'run_id':run_id,'policy_skipped':run.get('policy_skipped',0),'state':run['state'],'failures':run.get('failures',[]),'categories':categories,'results':rows,
            'identities':run['identities'],'manifest':manifest,'drift':current_drift,
            'comparison_rule':'Different fixture or rubric hashes are noncomparable. Route, constitution, template, and render changes indicate drift.',
            'markdown':'\n'.join(lines)}
