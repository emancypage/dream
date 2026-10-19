"""Stable dream-dpo-v1 data export; no model or database writes."""
import hashlib
import json
from character.constitution import load_constitution
from character.review import accepted_pairs
from character.safety import policy_for, check_constitution, policy_fingerprint
from character.render import manifest_for


def export_pairs(conn,config):
    document=load_constitution(config); policy=policy_for(config)
    check_constitution(document,policy)
    pairs=sorted(accepted_pairs(conn,config,document),key=lambda p:p['id'])
    lines=[]
    for pair in pairs:
        prompt=[{'role':m['role'],'content':m['content']} for m in pair['prompt_messages']]
        prompt_hash=hashlib.sha256(json.dumps(prompt,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        metadata={'format':'dream-dpo-v1','constitution_sha256':document.sha256,'constitution_version':document.version,
                  'pair_id':pair['id'],'pair_revision':pair['revision'],'pair_sha256':pair['content_sha256'],
                  'human_review_reason':pair['acceptance_reason'],'edited':bool(pair.get('edited',False)),
                  'review_event':pair.get('review_event',{}),'judgments':pair.get('judgments',[]),
                  'historical_overlap_warning':bool(pair.get('historical_overlap_warning',False)),
                  'same_family_override':bool(pair.get('same_family_override',False)),
                  'prompt_sha256':prompt_hash,'provenance':pair.get('provenance',{}),
                  'generator_calls':pair.get('generator_calls',[])}
        lines.append(json.dumps({'prompt':prompt,'chosen':pair['chosen'],'rejected':pair['rejected'],'metadata':metadata},ensure_ascii=False,sort_keys=True,separators=(',',':')))
    text='\n'.join(lines)+ ('\n' if lines else '')
    if policy.excludes(text=text):
        raise ValueError('serialized artifact conflicts with retention policy')
    manifest=manifest_for(document,policy,pairs,text)
    manifest['format']='dream-dpo-v1'
    return {'text':text,'manifest':manifest}
