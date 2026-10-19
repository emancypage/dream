"""Short startup examples with separate human approval and exact source binding."""
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sqlite3
import uuid

import backend
from model_types import validate_output
from character import storage
from character.types import canonical_json, content_hash
from character.constitution import load_constitution
from character.review import accepted_pairs
from character.safety import policy_for, check_constitution, policy_fingerprint

TABLE = 'character_short_examples'
FIELDS = ('situation', 'preferred', 'avoid', 'feedback', 'limits')
MAX_BODY_CODEPOINTS = 2500
PROMPT = Path(__file__).resolve().parent.parent / 'prompts' / 'character-short-v1.md'
SCHEMA = {'type':'object', 'properties':{key:{'type':'string'} for key in FIELDS},
          'required':list(FIELDS), 'additionalProperties':False}


def validate_body(body):
    if not isinstance(body, dict) or set(body) != set(FIELDS):
        raise ValueError('short example requires situation, preferred, avoid, feedback, limits')
    if any(not isinstance(body[key], str) or not body[key].strip() for key in FIELDS):
        raise ValueError('short example fields must be nonempty text')
    if sum(len(body[key]) for key in FIELDS) > MAX_BODY_CODEPOINTS:
        raise ValueError('short example exceeds 2500 codepoints')


def _context(config):
    doc = load_constitution(config)
    policy = policy_for(config)
    check_constitution(doc, policy)
    return doc, policy


def _source(conn, config, pair_id, sha, doc):
    pair = next((p for p in accepted_pairs(conn, config, doc) if p['id'] == pair_id), None)
    if pair is None or pair['content_sha256'] != sha:
        raise ValueError('accepted source changed or unavailable')
    return pair


def _rows(conn, config):
    return storage.eligible_records(conn, config, TABLE)


def list_examples(conn, config):
    return [show_example(conn, config, row['id']) for row in _rows(conn, config)]


def pending_review_count(conn, config):
    """Count current reviewable drafts without exposing any draft wording."""
    doc, _ = _context(config)
    sources = {p['id']:p['content_sha256'] for p in accepted_pairs(conn, config, doc)}
    return sum(row.get('status') == 'draft' and sources.get(row['pair_id']) == row['pair_sha256']
               for row in storage.eligible_records(conn, config, TABLE, doc))


def review_reminder(conn, config):
    count = pending_review_count(conn, config)
    if not count:
        return ''
    noun = 'draft is' if count == 1 else 'drafts are'
    return ('# Dream review reminder\nAt session start, briefly tell the user that '
            +str(count)+' short example '+noun+' waiting for review. '
            +'The drafts remain inactive until separate human approval. '
            +'Review command: dream character short list.\n')


def show_example(conn, config, identity):
    row = next((r for r in _rows(conn, config) if r['id'] == identity), None)
    if row is None:
        raise ValueError('short example unavailable under current policy')
    doc = load_constitution(config, require_approved=False)
    try:
        if row['constitution_sha256'] != doc.sha256:
            raise ValueError('changed constitution')
        _source(conn, config, row['pair_id'], row['pair_sha256'], doc)
    except ValueError:
        return dict(row, status='stale')
    return row


def draft_example(conn, config, pair_id, *, expected_pair_sha, body, expected_sha=None, generation=None):
    validate_body(body)
    doc, policy = _context(config)
    fingerprint = policy_fingerprint(policy)
    _source(conn, config, pair_id, expected_pair_sha, doc)
    if not storage._text_allowed(body, policy):
        raise ValueError('short example conflicts with policy')
    with storage.open_writer(config) as writer:
        writer.execute('BEGIN IMMEDIATE')
        current_doc, current_policy = _context(config)
        if current_doc.sha256 != doc.sha256 or policy_fingerprint(current_policy) != fingerprint:
            raise ValueError('inputs changed')
        pair = _source(writer, config, pair_id, expected_pair_sha, doc)
        existing = next((r for r in storage.records(writer, TABLE) if r['work_key'] == pair_id), None)
        if (existing is None and expected_sha is not None or
                existing is not None and existing['content_sha256'] != expected_sha):
            raise ValueError('short example changed; show it and provide its current hash')
        row = dict(id=existing['id'] if existing else 'short-'+uuid.uuid4().hex,
                   work_key=pair_id, revision=existing['revision']+1 if existing else 1,
                   format='dream-character-short-v1', status='draft', body=dict(body),
                   pair_id=pair_id, pair_sha256=expected_pair_sha, pair_revision=pair['revision'],
                   constitution_sha256=doc.sha256, constitution_version=doc.version,
                   created_at=existing['created_at'] if existing else datetime.now(timezone.utc).isoformat(),
                   updated_at=datetime.now(timezone.utc).isoformat())
        if generation is not None:
            row['generation'] = generation
        if not storage._text_allowed(row, current_policy):
            raise ValueError('short example conflicts with policy')
        if existing:
            return storage.update_record(writer, TABLE, row, expected_sha=expected_sha)
        return storage.insert_record(writer, TABLE, row)


def generate_example(conn, config, pair_id, *, expected_pair_sha):
    """Prepare unapproved wording without undoing the prior human acceptance."""
    if not config.data['character']['enabled']:
        return {'skipped':'character disabled'}
    try:
        doc, policy = _context(config)
        fingerprint = policy_fingerprint(policy)
        pair = _source(conn, config, pair_id, expected_pair_sha, doc)
        existing = next((r for r in storage.records(conn, TABLE) if r['work_key'] == pair_id), None)
        if existing is not None and existing['pair_sha256'] == expected_pair_sha:
            return {'skipped':'short example already exists for this source', 'id':existing['id']}
        template = PROMPT.read_text(encoding='utf-8')
        system = template + policy.context() + '\nApproved constitution:\n' + doc.text
        prompt = canonical_json({'messages':pair['prompt_messages'], 'preferred_answer':pair['chosen'],
                                 'avoided_answer':pair['rejected'], 'human_feedback':pair['acceptance_reason'],
                                 'trait_ids':pair['trait_ids']})
        started = datetime.now(timezone.utc).isoformat()
        answer = backend.generate('character_generate', prompt, SCHEMA, config=config, system_prompt=system)
        finished = datetime.now(timezone.utc).isoformat()
        validate_output(answer.output, SCHEMA)
        validate_body(answer.output)
        current_doc, current_policy = _context(config)
        if (current_doc.sha256 != doc.sha256 or policy_fingerprint(current_policy) != fingerprint
                or PROMPT.read_text(encoding='utf-8') != template):
            raise ValueError('short generation inputs changed')
        generation = dict(stage='character_generate', provider=config.stage('character_generate')['provider'],
                          configured_model=config.stage('character_generate').get('model'),
                          reported_model=getattr(answer,'model',None), options=dict(config.stage('character_generate')),
                          started_at=started, finished_at=finished, duration_ms=getattr(answer,'duration_ms',None),
                          usage=asdict(answer.usage) if is_dataclass(getattr(answer,'usage',None)) else None,
                          prompt_sha256=content_hash({'system':system,'prompt':prompt,'schema':SCHEMA}))
        return draft_example(conn, config, pair_id, expected_pair_sha=expected_pair_sha,
                             expected_sha=existing['content_sha256'] if existing else None,
                             body=answer.output, generation=generation)
    except (ValueError, RuntimeError, TimeoutError, OSError, sqlite3.Error) as exc:
        return draft_failure(pair_id, expected_pair_sha, exc)


def draft_failure(pair_id, expected_pair_sha, error):
    return {'failures':[{'pair_id':pair_id, 'error':type(error).__name__}],
            'retry':'dream character short generate '+pair_id+' --expected-pair-sha '+expected_pair_sha}


def review_example(conn, config, identity, action, *, expected_sha, reason):
    if action not in ('approve', 'reject'):
        raise ValueError('invalid short review action')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('human reason required')
    doc, policy = _context(config)
    fingerprint = policy_fingerprint(policy)
    initial = next((r for r in _rows(conn, config) if r['id'] == identity), None)
    if initial is None or initial['content_sha256'] != expected_sha:
        raise ValueError('short example changed or unavailable')
    with storage.open_writer(config) as writer:
        writer.execute('BEGIN IMMEDIATE')
        live_doc, live_policy = _context(config)
        if live_doc.sha256 != doc.sha256 or policy_fingerprint(live_policy) != fingerprint:
            raise ValueError('inputs changed')
        row = next((r for r in _rows(writer, config) if r['id'] == identity), None)
        if row is None or row['content_sha256'] != expected_sha:
            raise ValueError('short example changed or unavailable')
        if row['constitution_sha256'] != doc.sha256:
            raise ValueError('source constitution changed; draft again')
        _source(writer, config, row['pair_id'], row['pair_sha256'], doc)
        validate_body(row['body'])
        if not storage._text_allowed(reason, live_policy):
            raise ValueError('short review conflicts with policy')
        updated = storage.update_record(writer, TABLE, dict(row, revision=row['revision']+1,
                    status='approved' if action == 'approve' else 'rejected'), expected_sha=expected_sha)
        storage.insert_record(writer, 'character_review_events', dict(
            id='review-'+uuid.uuid4().hex, revision=1, action='short_'+action, origin='human_cli',
            short_id=identity, short_revision=updated['revision'], short_sha256=updated['content_sha256'],
            reviewed_sha256=expected_sha, pair_id=row['pair_id'], pair_sha256=row['pair_sha256'],
            reason=reason, constitution_sha256=doc.sha256, constitution_version=doc.version,
            created_at=datetime.now(timezone.utc).isoformat()))
        return updated


def approved_examples(conn, config, document=None):
    doc = document or load_constitution(config)
    pairs = {p['id']: p for p in accepted_pairs(conn, config, doc)}
    events = storage.eligible_records(conn, config, 'character_review_events', doc)
    selected = []
    for row in storage.eligible_records(conn, config, TABLE, doc):
        pair = pairs.get(row['pair_id'])
        if row['status'] != 'approved' or pair is None or pair['content_sha256'] != row['pair_sha256']:
            continue
        approval = next((e for e in events if e.get('action') == 'short_approve'
                         and e.get('origin') == 'human_cli' and e.get('short_id') == row['id']
                         and e.get('short_revision') == row['revision'] and e.get('short_sha256') == row['content_sha256']
                         and e.get('pair_sha256') == row['pair_sha256']), None)
        if approval is None:
            continue
        validate_body(row['body'])
        selected.append(dict(pair, short_example=row))
    return selected
