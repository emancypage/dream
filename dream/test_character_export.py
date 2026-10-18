import json
from test_character_render import inputs
from character import export,render

def test_exact_dpo_structure_stable_and_prompt_seq_removed(inputs,monkeypatch):
    config,pairs=inputs
    for name in ('load_constitution','policy_for','policy_fingerprint','accepted_pairs'):
        monkeypatch.setattr(export,name,getattr(render,name))
    first=export.export_pairs(None,config)
    assert first==export.export_pairs(None,config)
    lines=[json.loads(line) for line in first['text'].splitlines()]
    assert set(lines[0])=={'prompt','chosen','rejected','metadata'}
    assert lines[0]['prompt']==[{'role':'user','content':'Do this'}]
    assert lines[0]['metadata']['pair_id']=='a'
    assert lines[0]['metadata']['format']=='dream-dpo-v1'


def test_serialized_metadata_keys_checked_by_policy(inputs,monkeypatch):
    from types import SimpleNamespace
    import pytest
    config,pairs=inputs
    for name in ('load_constitution','accepted_pairs'):
        monkeypatch.setattr(export,name,getattr(render,name))
    monkeypatch.setattr(export,'policy_for',lambda c:SimpleNamespace(excludes=lambda **k:'"human_review_reason"' in k.get('text','')))
    with pytest.raises(ValueError,match='policy'):
        export.export_pairs(None,config)
