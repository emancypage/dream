from types import SimpleNamespace
from pathlib import Path
import os
import pytest
from config import load_config
from character import safety


def test_live_constitution_identity_alias(tmp_path):
    source=tmp_path/'rules'; source.write_text('rules')
    alias=tmp_path/'alias'; os.link(source,alias)
    assert safety.is_constitution_source(alias,constitution_path=source)


def test_artifact_rejects_stale_manifest(tmp_path,monkeypatch):
    cfg=load_config(tmp_path/'none',{'storage':{'memory_root':str(tmp_path/'memory'),'db_path':str(tmp_path/'db')},'character':{'state_root':str(tmp_path/'state'),'constitution_path':str(tmp_path/'rules')}})
    monkeypatch.setattr('character.constitution.load_constitution',lambda c:SimpleNamespace(sha256='new',text='Rules'))
    with pytest.raises(ValueError,match='changed'):
        safety.write_artifacts(cfg,{'block.md':'old'},{'constitution_sha256':'old','policy_fingerprint':safety.policy_fingerprint(safety.policy_for(cfg)),'pair_ids':[],'pair_revisions':{},'pair_hashes':{}})
    assert not (tmp_path/'state'/'block.md').exists()


def test_policy_changed_after_render_rejected(tmp_path,monkeypatch):
    cfg=load_config(tmp_path/'none',{'storage':{'memory_root':str(tmp_path/'memory'),'db_path':str(tmp_path/'db')},'character':{'state_root':str(tmp_path/'state'),'constitution_path':str(tmp_path/'rules')}})
    monkeypatch.setattr('character.constitution.load_constitution',lambda c:SimpleNamespace(sha256='same',text='Rules'))
    fingerprint=safety.policy_fingerprint(safety.policy_for(cfg))
    cfg.memory_root.mkdir(); (cfg.memory_root/'.dream-retention.json').write_text('{"text_markers":["secret"]}')
    with pytest.raises(ValueError,match='changed'):
        safety.write_artifacts(cfg,{'block.md':'Rules'},{'constitution_sha256':'same','policy_fingerprint':fingerprint,'pair_ids':[]})


def test_output_backup_and_replacement(tmp_path,monkeypatch):
    cfg=load_config(tmp_path/'none',{'storage':{'memory_root':str(tmp_path/'memory'),'db_path':str(tmp_path/'db')},'character':{'state_root':str(tmp_path/'state'),'constitution_path':str(tmp_path/'rules')}})
    monkeypatch.setattr('character.constitution.load_constitution',lambda c:SimpleNamespace(sha256='same',text='Rules'))
    manifest={'constitution_sha256':'same','policy_fingerprint':safety.policy_fingerprint(safety.policy_for(cfg)),'pair_ids':[]}
    safety.write_artifacts(cfg,{'block.md':'old'},manifest)
    safety.write_artifacts(cfg,{'block.md':'new'},manifest)
    assert (tmp_path/'state'/'block.md').read_text()=='new'
    assert [p.read_text() for p in (tmp_path/'state'/'backups').iterdir()]==['old']


def test_custom_constitution_excluded_from_sync_and_consolidation(tmp_path):
    import sqlite3
    from recall_documents import synchronize_recall_documents
    from consolidate import read_current_memory
    root=tmp_path/'memory'; root.mkdir()
    source=tmp_path/'custom-rules'; source.write_text('private constitution marker')
    os.link(source,root/'alias.md')
    (root/'notes.md').write_text('ordinary memory')
    conn=sqlite3.connect(':memory:'); conn.executescript((Path(__file__).parent/'schema.sql').read_text())
    synchronize_recall_documents(conn,root,include_raw_transcripts=False,constitution_path=source)
    assert not any('constitution marker' in row[0] for row in conn.execute('SELECT text FROM recall_documents'))
    text=read_current_memory(root,constitution_path=source)
    assert 'constitution marker' not in text and 'ordinary memory' in text


def test_already_indexed_constitution_alias_blocked_at_read(tmp_path):
    import sqlite3
    from recall_documents import synchronize_recall_documents
    from recall_context import run_context
    from recall_types import RecallQuery
    root=tmp_path/'memory'; root.mkdir()
    source=tmp_path/'custom-rules'; source.write_text('Postgres migration private constitution')
    os.link(source,root/'alias.md')
    conn=sqlite3.connect(':memory:'); conn.executescript((Path(__file__).parent/'schema.sql').read_text())
    synchronize_recall_documents(conn,root,include_raw_transcripts=False,constitution_path=None)
    assert conn.execute('SELECT count(*) FROM recall_documents').fetchone()[0]==1
    cfg=load_config(tmp_path/'none')
    query=RecallQuery('Postgres migration','s','session-start:compact',str(root),(),6000,frozenset(),False)
    result=run_context(conn,query,cfg.recall_settings(),memory_root=root,constitution_path=source)
    assert result.selected==() and 'private constitution' not in result.rendered_context


from test_character_judge import env


def test_pair_edited_after_pure_render_cannot_be_written(env,monkeypatch):
    from test_character_review import judged
    from character import render,storage
    j,r,c,db,p=judged(env,monkeypatch)
    document=r.load_constitution(c)
    monkeypatch.setattr(render,'load_constitution',lambda config:document)
    monkeypatch.setattr('character.constitution.load_constitution',lambda config:document)
    accepted=r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked')
    from test_character_short import approve_example
    approve_example(db,c,accepted)
    result=render.render_block(db,c)
    storage.update_record(db,'character_pairs',dict(accepted,revision=accepted['revision']+1,chosen='Edited later'),expected_sha=accepted['content_sha256']); db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    with pytest.raises(ValueError,match='pair changed'):
        safety.write_artifacts(c,{'block.md':result['text']},result['manifest'])
    assert not (Path(c.data['character']['state_root'])/'block.md').exists()


def _accepted_with_parent(env,monkeypatch,parent_kind):
    from test_character_review import judged
    from character import storage,review,render,export
    j,r,c,db,p=env
    table='character_moments' if parent_kind=='moment' else 'character_seeds'
    parent=storage.insert_record(db,table,{'id':'moment-parent','revision':1,'constitution_sha256':'abc','disposition':'completed','prompt_messages':[],'extraction':{'family':'openai'}})
    p=storage.update_record(db,'character_pairs',dict(p,revision=2,**{parent_kind+'_id':parent['id']},provenance={'source_content_sha256':parent['content_sha256']}),expected_sha=p['content_sha256']); db.commit()
    judged((j,r,c,db,p),monkeypatch)
    accepted=r.review_pair(db,c,'p','accept',expected_sha=p['content_sha256'],reason='Checked source')
    document=r.load_constitution(c)
    monkeypatch.setattr(render,'load_constitution',lambda config:document)
    monkeypatch.setattr(export,'load_constitution',lambda config:document)
    monkeypatch.setattr('character.constitution.load_constitution',lambda config:document)
    from test_character_short import approve_example
    approve_example(db,c,accepted)
    return c,db,parent,accepted


@pytest.mark.parametrize('parent_kind',['moment','seed'])
def test_changed_source_parent_excluded_from_render_and_export(env,monkeypatch,parent_kind):
    from character import storage,render,export
    c,db,parent,pair=_accepted_with_parent(env,monkeypatch,parent_kind)
    storage.update_record(db,'character_moments' if parent_kind=='moment' else 'character_seeds',dict(parent,revision=2,note='Changed source'),expected_sha=parent['content_sha256']); db.commit()
    assert not render.render_block(db,c)['manifest']['pair_ids']
    assert export.export_pairs(db,c)['text']==''


@pytest.mark.parametrize('parent_kind',['moment','seed'])
def test_changed_source_after_preview_cannot_be_written(env,monkeypatch,parent_kind):
    from character import storage,render
    c,db,parent,pair=_accepted_with_parent(env,monkeypatch,parent_kind)
    result=render.render_block(db,c)
    assert result['manifest']['pair_ids']==['p']
    storage.update_record(db,'character_moments' if parent_kind=='moment' else 'character_seeds',dict(parent,revision=2,note='Changed source'),expected_sha=parent['content_sha256']); db.commit()
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    with pytest.raises(ValueError,match='pair changed'):
        safety.write_artifacts(c,{'block.md':result['text']},result['manifest'])
