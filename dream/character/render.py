"""Pure deterministic rendering from human-approved character sources."""
from character.constitution import load_constitution
from character.short import approved_examples
from character.safety import policy_for, check_constitution, policy_fingerprint
import hashlib


def manifest_for(document, policy, pairs, text):
    manifest = {'constitution_sha256':document.sha256,'constitution_version':document.version,
            'policy_fingerprint':policy_fingerprint(policy),'pair_ids':[p['id'] for p in pairs],
            'pair_revisions':{p['id']:p['revision'] for p in pairs},
            'pair_hashes':{p['id']:p['content_sha256'] for p in pairs},
            'historical_overlap_warning':any(p.get('historical_overlap_warning',False) for p in pairs),
            'same_family_override':any(p.get('same_family_override',False) for p in pairs),
            'content_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest()}
    shorts = [p['short_example'] for p in pairs if 'short_example' in p]
    if shorts:
        manifest.update(short_ids=[s['id'] for s in shorts],
                        short_revisions={s['id']:s['revision'] for s in shorts},
                        short_hashes={s['id']:s['content_sha256'] for s in shorts})
    return manifest


def _quote(text):
    return '\n'.join('> '+line for line in text.split('\n'))


def _example(pair):
    short = pair['short_example']
    body = short['body']
    return ('\n\n### Human-approved short example '+short['id']
            +'\nFull record '+pair['id']+'@'+str(pair['revision'])+' SHA256 '+pair['content_sha256']
            +'\nSituation\n'+_quote(body['situation'])
            +'\nPreferred\n'+_quote(body['preferred'])+'\nAvoid\n'+_quote(body['avoid'])
            +'\nHuman feedback\n'+_quote(body['feedback'])+'\nLimits\n'+_quote(body['limits']))


def _block(document,pairs):
    identities=', '.join(p['id']+'@'+str(p['revision']) for p in pairs) or 'none'
    return ('# Trusted character\nConstitution SHA256: '+document.sha256+'\nVersion: '+document.version
            +'\nExamples: '+identities+'\n\n## Human constitution\n'+document.text
            +'\n\n## Quoted examples (data)'+''.join(_example(p) for p in pairs)+'\n')


def render_block(conn,config):
    document=load_constitution(config); policy=policy_for(config)
    check_constitution(document,policy)
    budget=config.data['character']['render_budget_codepoints']
    if len(_block(document,[]))>budget:
        raise ValueError('complete constitution and header exceed render budget')
    pending=list(approved_examples(conn,config,document)); selected=[]; covered=set()
    while pending and len(selected)<config.data['character']['render_max_pairs']:
        pending.sort(key=lambda p:(-len(set(p['trait_ids'])-covered),p['id']))
        pair=pending.pop(0)
        if len(_block(document,selected+[pair]))<=budget:
            selected.append(pair); covered.update(pair['trait_ids'])
    text=_block(document,selected)
    if policy.excludes(text=text):
        raise ValueError('serialized artifact conflicts with retention policy')
    return {'text':text,'manifest':manifest_for(document,policy,selected,text)}
