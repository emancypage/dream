"""Local human review with revision and current-policy checks."""
from datetime import datetime, timezone
import uuid
from character import storage
from character.constitution import load_constitution
from character.safety import policy_for, check_constitution, policy_fingerprint
from character.types import content_hash
from config import provider_family


def candidate_hash(pair):
    return content_hash({key:pair.get(key) for key in ('prompt_messages','chosen','rejected','trait_ids','constitution_sha256')})


def _context(config, document=None):
    doc=document or load_constitution(config)
    policy=policy_for(config)
    check_constitution(doc,policy)
    return doc,policy


def check_source_parent(conn, config, pair, document):
    """Require displayed evidence to match its recorded source identity."""
    parent_id=pair.get('moment_id') or pair.get('seed_id')
    if parent_id is None:
        return None
    table='character_moments' if pair.get('moment_id') else 'character_seeds'
    parent=next((row for row in storage.eligible_records(conn,config,table,document) if row['id']==parent_id),None)
    if parent is None or parent['content_sha256']!=pair.get('provenance',{}).get('source_content_sha256'):
        raise ValueError('source evidence changed or unavailable')
    return parent


def _pairs(conn,config,doc):
    return storage.eligible_records(conn,config,'character_pairs',doc)


def _judgments(conn,config,pair,doc,*,exact=True):
    return [j for j in storage.eligible_records(conn,config,'character_judgments',doc)
            if j.get('pair_id')==pair['id'] and j.get('complete') is True
            and j.get('candidate_sha256')==candidate_hash(pair)
            and (not exact or j.get('pair_revision')==pair['revision'] and j.get('pair_sha256')==pair['content_sha256'])]


def list_pairs(conn,config):
    doc=load_constitution(config,require_approved=False)
    return [show_pair(conn,config,p['id']) for p in storage.eligible_records(conn,config,'character_pairs')]


def show_pair(conn,config,id):
    doc=load_constitution(config,require_approved=False)
    pair=next((p for p in storage.eligible_records(conn,config,'character_pairs') if p['id']==id),None)
    if pair is None: raise ValueError('pair unavailable under current policy')
    judgments=[j for j in storage.eligible_records(conn,config,'character_judgments') if j.get('pair_id')==id]
    events=[e for e in storage.eligible_records(conn,config,'character_review_events') if e.get('pair_id')==id]
    history=pair.get('provenance',{}).get('historical_family','anthropic' if pair.get('source')=='claude' or pair.get('provenance',{}).get('source')=='claude' else 'unknown')
    overlap=history==provider_family(config,'character_judge')
    status='stale' if pair.get('constitution_sha256')!=doc.sha256 and pair['status'] in ('pending','deferred','edited') else pair['status']
    return dict(pair,status=status,judgments=judgments,review_events=events,historical_overlap_warning=overlap or any(j.get('historical_overlap_warning') for j in judgments))


def accepted_pairs(conn,config,document=None):
    doc,_=_context(config,document)
    events=storage.eligible_records(conn,config,'character_review_events',doc)
    accepted=[]
    for pair in _pairs(conn,config,doc):
        if pair.get('status')!='accepted' or pair.get('accepted_revision')!=pair['revision'] or pair.get('accepted_candidate_sha256')!=candidate_hash(pair): continue
        event=next((e for e in events if e.get('pair_id')==pair['id'] and e.get('action')=='accept' and e.get('origin')=='human_cli' and e.get('accepted_revision')==pair['revision'] and e.get('candidate_sha256')==candidate_hash(pair) and e.get('answer')==pair.get('accepted_answer')),None)
        if event is None: continue
        try:
            check_source_parent(conn,config,pair,doc)
        except ValueError:
            continue
        judgments=_judgments(conn,config,pair,doc,exact=False)
        oriented=dict(pair)
        if pair.get('accepted_answer')=='right':
            oriented.update(chosen=pair['rejected'],rejected=pair['chosen'])
        accepted.append(dict(oriented,acceptance_reason=event['reason'],review_event=event,judgments=judgments,historical_overlap_warning=any(j.get('historical_overlap_warning') for j in judgments),same_family_override=any(j.get('same_family_override') for j in judgments)))
    return accepted


def review_pair(conn,config,id,action,*,expected_sha,reason='',body=None,acknowledge_flags=False,answer='left'):
    if action not in ('accept','reject','defer','edit'): raise ValueError('invalid review action')
    if answer not in ('left','right') or action!='accept' and answer!='left': raise ValueError('invalid answer selection')
    if action in ('accept','reject','edit') and (not isinstance(reason,str) or not reason.strip()): raise ValueError('human reason required')
    if action=='edit' and (not isinstance(body,str) or not body.strip() or len(body)>config.data['character']['max_response_codepoints']): raise ValueError('valid chosen body required')
    doc,policy=_context(config); fingerprint=policy_fingerprint(policy)
    initial=next((p for p in _pairs(conn,config,doc) if p['id']==id),None)
    if initial is None or initial['content_sha256']!=expected_sha: raise ValueError('pair changed or unavailable')
    if initial.get('status')=='stale': raise ValueError('stale pair requires generation refresh')
    if action=='accept': check_source_parent(conn,config,initial,doc)
    if not storage._text_allowed({'body':body,'reason':reason},policy): raise ValueError('review conflicts with policy')
    with storage.open_writer(config) as writer:
        writer.execute('BEGIN IMMEDIATE')
        current_doc,current_policy=_context(config)
        if current_doc.sha256!=doc.sha256 or policy_fingerprint(current_policy)!=fingerprint: raise ValueError('inputs changed')
        pair=next((p for p in _pairs(writer,config,doc) if p['id']==id),None)
        if pair is None or pair['content_sha256']!=expected_sha: raise ValueError('pair changed or unavailable')
        if pair.get('status')=='stale': raise ValueError('stale pair requires generation refresh')
        flags=[]
        if action=='accept':
            check_source_parent(writer,config,pair,doc)
            if pair['status'] in ('rejected','stale'): raise ValueError('pair state cannot be accepted')
            judgments=_judgments(writer,config,pair,doc)
            if not judgments: raise ValueError('current complete judgment required')
            if pair.get('conflict'): flags.append('conflict')
            if any(j['normalized_result']=='inconsistent' for j in judgments): flags.append('inconsistent')
            selected='chosen' if answer=='left' else 'rejected'
            if any(j['normalized_result']!=selected for j in judgments): flags.append('against_chosen' if answer=='left' else 'against_selected')
            if flags and not acknowledge_flags: raise ValueError('explicit flag acknowledgement required')
        updated=dict(pair,revision=pair['revision']+1,status={'accept':'accepted','reject':'rejected','defer':'deferred','edit':'edited'}[action])
        for key in ('accepted_revision','accepted_candidate_sha256','accepted_at','accepted_answer'): updated.pop(key,None)
        if action=='edit':
            if body.strip()==pair['rejected'].strip(): raise ValueError('equal responses')
            updated.update(chosen=body,edited=True,judgment_ids=[])
        if action=='accept': updated.update(accepted_revision=updated['revision'],accepted_candidate_sha256=candidate_hash(updated),accepted_at=datetime.now(timezone.utc).isoformat(),accepted_answer=answer)
        updated=storage.update_record(writer,'character_pairs',updated,expected_sha=expected_sha)
        storage.insert_record(writer,'character_review_events',dict(id='review-'+uuid.uuid4().hex,revision=1,pair_id=id,pair_revision=pair['revision'],pair_sha256=expected_sha,candidate_sha256=candidate_hash(updated),accepted_revision=updated['revision'] if action=='accept' else None,answer=answer if action=='accept' else None,action=action,reason=reason,origin='human_cli',old_status=pair['status'],new_status=updated['status'],acknowledged_flags=flags if acknowledge_flags else [],created_at=datetime.now(timezone.utc).isoformat(),constitution_sha256=doc.sha256,constitution_version=doc.version,prompt_version=pair.get('prompt_version'),provenance=pair.get('provenance',{})))
        storage.mark_stale(writer,doc)
        return updated
