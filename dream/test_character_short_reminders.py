"""Review notices reveal counts, while draft text stays outside session context."""
import io
import json
from types import SimpleNamespace
import pytest
import dream
import metrics
from config import load_config
from test_character_judge import env
from test_character_short import BODY, prepared, approve_example
from character import short, storage
from recall_context import compose_hook_context
from recall_types import RecallDiagnostics, RecallResult


def test_pending_count_tracks_review_and_ignores_stale_or_excluded_sources(env, monkeypatch):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    assert short.pending_review_count(db,c) == 0
    draft = short.draft_example(db,c,'p',expected_pair_sha=pair['content_sha256'],body=BODY)
    assert short.pending_review_count(db,c) == 1
    approved = short.review_example(db,c,draft['id'],'approve',expected_sha=draft['content_sha256'],reason='Checked')
    assert short.pending_review_count(db,c) == 0
    draft = short.draft_example(db,c,'p',expected_pair_sha=pair['content_sha256'],body=BODY,expected_sha=approved['content_sha256'])
    storage.update_record(db,'character_pairs',dict(pair,revision=pair['revision']+1,status='rejected'),expected_sha=pair['content_sha256']);db.commit()
    assert short.pending_review_count(db,c) == 0


def test_pending_count_excludes_policy_hidden_drafts(env, monkeypatch):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    short.draft_example(db,c,'p',expected_pair_sha=pair['content_sha256'],body=BODY)
    c.memory_root.mkdir()
    (c.memory_root/'.dream-retention.json').write_text(json.dumps({'text_markers':[BODY['feedback']]}))
    assert short.pending_review_count(db,c) == 0


@pytest.mark.parametrize('injection',[True,False])
def test_session_start_reminds_without_injecting_draft_text(env, monkeypatch, capsys, injection):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    c.data['character']['hook_injection'] = injection
    draft = short.draft_example(db,c,'p',expected_pair_sha=pair['content_sha256'],body=BODY)
    monkeypatch.setattr(dream,'claim_hook_event',lambda *a:'token')
    monkeypatch.setattr(dream,'finish_hook_event',lambda *a,**k:None)
    monkeypatch.setattr(dream,'run_context',lambda *a,**k:RecallResult((),'normal recall','lexical',None,RecallDiagnostics(0,0,None,0,0,None)))
    monkeypatch.setattr('backend.generate',lambda *a,**k:pytest.fail('reminders cannot call a model'))
    def context(event):
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        payload={'session_id':'review-reminder-test','source':'startup','cwd':str(c.db_path.parent),'prompt':'Continue'}
        monkeypatch.setattr(dream.sys,'stdin',io.StringIO(json.dumps(payload)))
        assert dream.cmd_context(SimpleNamespace(context_cmd=event,config=c,db=c.db_path,explain=False)) == 0
        return json.loads(capsys.readouterr().out)['hookSpecificOutput']['additionalContext']
    text=context('session-start')
    assert '1 short example draft' in text and 'dream character short list' in text
    assert 'inactive' in text and 'normal recall' in text
    assert BODY['feedback'] not in text and BODY['preferred'] not in text
    assert 'short example draft' not in context('prompt')
    short.review_example(db,c,draft['id'],'approve',expected_sha=draft['content_sha256'],reason='Approved')
    assert 'Dream review reminder' not in context('session-start')


def test_reminder_survives_oversized_character_without_breaking_budget():
    text,diagnostic=compose_hook_context('memory','x'*1000,200,200,reminder_text='Review 2 inactive drafts.')
    assert text.startswith('Review 2 inactive drafts.') and 'memory' in text
    assert 'xxx' not in text and diagnostic=='character-over-budget'
    assert len(text.encode())<=200


def test_metrics_scrape_reads_current_pending_count_and_approval_without_private_text(env, monkeypatch, tmp_path):
    short_module,r,c,db,pair = prepared(env,monkeypatch)
    draft=short.draft_example(db,c,'p',expected_pair_sha=pair['content_sha256'],body=BODY)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    text=metrics.collect_metrics(tmp_path/'run.json',config=c)
    assert 'dream_character_short_drafts_pending 1\n' in text
    assert 'dream_character_short_review_source_up 1\n' in text
    assert BODY['feedback'] not in text and draft['id'] not in text and str(c.db_path) not in text
    short.review_example(db,c,draft['id'],'approve',expected_sha=draft['content_sha256'],reason='Approved')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    assert 'dream_character_short_drafts_pending 0\n' in metrics.collect_metrics(tmp_path/'run.json',config=c)


def test_metric_read_failure_omits_count_instead_of_reporting_zero(tmp_path):
    c=load_config(tmp_path/'none',{'storage':{'db_path':str(tmp_path/'missing-db')}})
    text=metrics.collect_metrics(tmp_path/'run.json',config=c)
    assert 'dream_run_health' in text
    assert 'dream_character_short_review_source_up 0\n' in text
    assert not any(line.startswith('dream_character_short_drafts_pending ') for line in text.splitlines())
    assert not c.db_path.exists()
