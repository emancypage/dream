from recall_context import compose_hook_context
import pytest

def test_complete_character_and_typed_recall():
    text,reason=compose_hook_context('memory','rules',1000,1000)
    assert text=='rules\n\n# Untrusted recall\nmemory' and reason is None

def test_omit_full_character_and_preserve_recall():
    text,reason=compose_hook_context('memory','x'*100,60,60)
    assert 'memory' in text and 'x' not in text and reason=='character-over-budget'

def test_conservative_unicode_bytes_bound():
    text,reason=compose_hook_context('界'*100,'rules',100,60)
    assert len(text.encode('utf-8'))<=60 and text.startswith('rules')


@pytest.mark.parametrize('generation_enabled', [True, False])
def test_compact_hook_separates_trusted_and_recall(tmp_path,monkeypatch,capsys,generation_enabled):
    import io,json
    from types import SimpleNamespace
    import dream
    from config import load_config
    from recall_types import RecallDiagnostics,RecallResult
    from character import render
    cfg=load_config(tmp_path/'none',{'storage':{'db_path':str(tmp_path/'db'),'memory_root':str(tmp_path/'memory')},'character':{'enabled':generation_enabled,'hook_injection':True,'constitution_path':str(tmp_path/'rules'),'state_root':str(tmp_path/'state')}})
    monkeypatch.setattr(dream.sys,'stdin',io.StringIO(json.dumps({'session_id':'compact-session','cwd':str(tmp_path),'source':'compact'})))
    monkeypatch.setattr(dream,'open_db_readonly',lambda path:object())
    monkeypatch.setattr(dream,'claim_hook_event',lambda *a:'token')
    monkeypatch.setattr(dream,'finish_hook_event',lambda *a,**k:None)
    seen=[]
    def recalled(conn,query,settings,**kwargs):
        seen.append((query,kwargs))
        return RecallResult((),'untrusted attack','lexical',None,RecallDiagnostics(0,0,None,16,0,None))
    monkeypatch.setattr(dream,'run_context',recalled)
    monkeypatch.setattr(render,'render_block',lambda conn,config:{'text':'trusted rules','manifest':{}})
    assert dream.cmd_context(SimpleNamespace(context_cmd='session-start',config=cfg,db=cfg.db_path,explain=False))==0
    text=json.loads(capsys.readouterr().out)['hookSpecificOutput']['additionalContext']
    assert text=='trusted rules\n\n# Untrusted recall\nuntrusted attack'
    assert seen[0][0].hook_event=='session-start:compact'
    assert seen[0][1]['memory_root']==cfg.memory_root


def test_no_character_preserves_normal_recall_unchanged():
    recall='memory '*1000
    assert compose_hook_context(recall,'',6000,1800)==(recall,None)
    assert compose_hook_context(recall,'rules'*1000,6000,1800)==(recall,'character-over-budget')
