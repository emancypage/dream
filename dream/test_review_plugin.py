"""The local panel delegates human decisions to Dream's existing review rules."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_character_judge import env
from test_character_review import judged
from test_character_short import BODY
from character import storage, short

PLUGIN = Path(__file__).resolve().parents[1] / 'plugins' / 'dream-review'


@pytest.fixture
def app(env, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('dream_review_app', PLUGIN / 'review_app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    j, r, cfg, conn, pair = env
    monkeypatch.setenv('DREAM_CONTROL_STATE_DIR', str(tmp_path / 'control'))
    monkeypatch.setattr(module, 'load_constitution', r.load_constitution)
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    instance = module.ReviewApp(cfg)
    connect = instance._connect
    def checkpointed_reader():
        # The fixture keeps a writer open for assertions. Production readers
        # deliberately refuse an active WAL; checkpoint fixture writes first.
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return connect()
    monkeypatch.setattr(instance, '_connect', checkpointed_reader)
    return instance


def pair_decision(pair, **extra):
    return dict(kind='pair', id=pair['id'], expected_sha=pair['content_sha256'],
                choice='left', reason='I prefer this answer.', acknowledge_flags=False, **extra)


def test_queue_reads_full_evidence_without_changing_data(app, env):
    j, r, cfg, conn, pair = env
    before = storage.snapshot(conn)
    result = app.queue()
    assert result['counts']['pairs'] == 1
    assert result['pairs'][0]['id'] == pair['id']
    detail = app.detail('pair', pair['id'])
    assert detail['chosen'] == 'Short' and detail['rejected'] == 'Long'
    assert detail['prompt_messages'] == pair['prompt_messages']
    assert storage.snapshot(conn) == before


def test_pair_choice_saved_and_short_stays_unapproved(app, env, monkeypatch):
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    result = app.decide(**pair_decision(pair))
    assert result['saved'] is True and result['record']['status'] == 'accepted'
    saved = storage.get_record(conn, 'character_pairs', pair['id'])
    assert saved['accepted_answer'] == 'left'
    draft = storage.records(conn, 'character_short_examples')[0]
    assert draft['status'] == 'draft'
    assert app.queue()['counts']['shorts'] == 1
    assert app.detail('short', draft['id'])['body'] == BODY


def test_right_choice_needs_explicit_acknowledgment(app, env, monkeypatch):
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    args = pair_decision(pair)
    args['choice'] = 'right'
    with pytest.raises(ValueError, match='acknowledgement'):
        app.decide(**args)
    args['acknowledge_flags'] = True
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    assert app.decide(**args)['record']['accepted_answer'] == 'right'


def test_stale_hash_reason_and_double_save_do_not_write(app, env):
    j, r, cfg, conn, pair = env
    args = pair_decision(pair)
    args['choice'] = 'reject'
    args['expected_sha'] = '0' * 64
    with pytest.raises(ValueError):
        app.decide(**args)
    args['expected_sha'] = pair['content_sha256']
    args['reason'] = ''
    with pytest.raises(ValueError, match='reason'):
        app.decide(**args)
    args['reason'] = 'Neither answer helps.'
    assert app.decide(**args)['record']['status'] == 'rejected'
    with pytest.raises(ValueError):
        app.decide(**args)
    assert len(storage.records(conn, 'character_review_events')) == 1


def test_short_edit_is_a_draft_and_needs_new_hash_for_approval(app, env, monkeypatch):
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    app.decide(**pair_decision(pair))
    draft = storage.records(conn, 'character_short_examples')[0]
    changed = app.edit_short(draft['id'], draft['content_sha256'], draft['pair_sha256'],
                             dict(BODY, preferred='Check the actual result.'))
    assert changed['record']['status'] == 'draft'
    with pytest.raises(ValueError):
        app.decide('short', draft['id'], draft['content_sha256'], 'approve', 'Reviewed')
    result = app.decide('short', draft['id'], changed['record']['content_sha256'], 'approve', 'Reviewed the exact wording')
    assert result['record']['status'] == 'approved'
    assert app.queue()['counts']['shorts'] == 0
    assert app.detail('short', draft['id'])['review_events'][-1]['reason'] == 'Reviewed the exact wording'


def test_short_generation_failure_does_not_undo_the_human_choice(app, env, monkeypatch):
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    def fail(*a, **k):
        raise RuntimeError('Provider unavailable')
    monkeypatch.setattr(j.backend, 'generate', fail)
    result = app.decide(**pair_decision(pair))
    assert result['saved'] is True
    assert result['short_draft']['failures']
    assert storage.get_record(conn, 'character_pairs', pair['id'])['status'] == 'accepted'
    accepted = app.detail('pair', pair['id'])
    assert accepted['can_prepare_short'] is True
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    retried = app.prepare_short(pair['id'], accepted['content_sha256'])
    assert retried['short_draft']['status'] == 'draft'
    assert app.detail('pair', pair['id'])['can_prepare_short'] is False


def test_reader_error_after_commit_reports_saved_choice(app, env, monkeypatch):
    import sqlite3
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    connect = app._connect
    calls = 0
    def reader():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise sqlite3.OperationalError('database busy')
        return connect()
    monkeypatch.setattr(app, '_connect', reader)
    result = app.decide(**pair_decision(pair))
    assert result['saved'] is True and result['short_draft']['failures']
    assert storage.get_record(conn, 'character_pairs', pair['id'])['status'] == 'accepted'


def test_pause_blocks_reads_and_writes_before_access(app, env, monkeypatch):
    import pause_control
    pause_control.pause()
    monkeypatch.setattr(app, '_connect', lambda: pytest.fail('database access while paused'))
    with pytest.raises(ValueError, match='Dream paused'):
        app.queue()
    with pytest.raises(ValueError, match='Dream paused'):
        app.decide(**pair_decision(env[-1]))


def test_pause_during_generation_discards_the_draft(app, env, monkeypatch):
    import pause_control
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    def pause_then_return(*a, **k):
        pause_control.pause()
        return SimpleNamespace(output=BODY)
    monkeypatch.setattr(j.backend, 'generate', pause_then_return)
    result = app.decide(**pair_decision(pair))
    assert result['saved'] is True
    assert result['short_draft']['failures']
    assert storage.records(conn, 'character_short_examples') == []
    assert storage.get_record(conn, 'character_pairs', pair['id'])['status'] == 'accepted'


def test_retention_excludes_private_pair(app, env):
    j, r, cfg, conn, pair = env
    cfg.memory_root.mkdir()
    (cfg.memory_root / '.dream-retention.json').write_text('{"text_markers":["Long"]}')
    assert app.queue()['pairs'] == []
    with pytest.raises(ValueError):
        app.detail('pair', pair['id'])


def test_stale_pair_is_visible_but_cannot_be_reviewed(app, env):
    j, r, cfg, conn, pair = env
    old = storage.insert_record(conn, 'character_pairs', dict(pair, id='old', constitution_sha256='old'))
    conn.commit()
    assert app.queue()['counts']['stale_pairs'] == 1
    assert app.detail('pair', 'old')['status'] == 'stale'
    args = pair_decision(old)
    args['choice'] = 'reject'
    with pytest.raises(ValueError):
        app.decide(**args)


@pytest.fixture
def server(app, monkeypatch):
    monkeypatch.syspath_prepend(str(PLUGIN))
    spec = importlib.util.spec_from_file_location('dream_review_server', PLUGIN / 'server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ReviewServer(app)


def test_only_open_tool_is_exposed_to_model(server):
    specs = server.tools()
    public = [t['name'] for t in specs if 'model' in t['_meta']['ui']['visibility']]
    assert public == ['show_review']
    opening = specs[0]
    assert opening['_meta']['openai/ui']['entrypoints'] == [{'type': 'global'}, {'type': 'thread'}]
    assert opening['_meta']['ui']['resourceUri'].startswith('ui://')
    import base64
    icon = opening['icons'][0]
    assert icon['mimeType'] == 'image/svg+xml'
    assert icon['src'].startswith('data:image/svg+xml;base64,')
    assert base64.b64decode(icon['src'].split(',', 1)[1]) == (PLUGIN / 'assets' / 'icon.svg').read_bytes()


def test_decisions_require_the_exact_displayed_hash_and_token(server, env):
    pair = env[-1]
    args = dict(kind='pair', id=pair['id'], expected_sha=pair['content_sha256'],
                choice='reject', reason='Neither helps.', acknowledge_flags=False, view_token='forged')
    with pytest.raises(ValueError, match='display'):
        server.call('save_decision', args)
    shown = server.call('get_item', {'kind': 'pair', 'id': pair['id']})
    args['view_token'] = shown['view_token']
    assert server.call('save_decision', args)['saved'] is True
    with pytest.raises(ValueError):
        server.call('save_decision', args)


@pytest.mark.parametrize('args', [
    {'kind': 'pair', 'id': 'p', 'extra': True},
    {'kind': 'pair', 'id': 42},
    {'kind': 'unknown', 'id': 'p'},
])
def test_server_validates_app_arguments(server, args):
    with pytest.raises(ValueError):
        server.call('get_item', args)


def test_protocol_discovery_resource_and_errors(server):
    initialized = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                                 'params': {'protocolVersion': '2025-11-25'}})
    assert initialized['result']['capabilities']['resources'] == {}
    resource = server.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'resources/read',
                              'params': {'uri': server.uri}})
    assert 'Dream Review' in resource['result']['contents'][0]['text']
    assert resource['result']['contents'][0]['mimeType'] == 'text/html;profile=mcp-app'
    bad = server.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
                         'params': {'name': 'save_decision', 'arguments': {}}})
    assert bad['result']['isError'] is True
    assert server.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None


def test_delete_pair_without_a_choice_or_reason_removes_evidence(app, env, monkeypatch):
    j, r, cfg, conn, pair = judged(env, monkeypatch)
    monkeypatch.setattr(j.backend, 'generate', lambda *a, **k: SimpleNamespace(output=BODY))
    accepted = app.decide(**pair_decision(pair))['record']
    result = app.delete_pair(pair['id'], accepted['content_sha256'])
    assert result == {'saved': True, 'deleted': True}
    assert storage.records(conn, 'character_pairs') == []
    assert storage.records(conn, 'character_judgments') == []
    assert storage.records(conn, 'character_short_examples') == []
    assert app.queue()['history'] == []
    events = storage.records(conn, 'character_review_events')
    assert len(events) == 1 and events[0]['action'] == 'delete'
    assert 'Task' not in str(events) and 'Short' not in str(events)
    with pytest.raises(ValueError):
        app.detail('pair', pair['id'])


def test_delete_requires_current_hash_and_view_token(server, env):
    pair = env[-1]
    args = dict(id=pair['id'], expected_sha=pair['content_sha256'], view_token='missing')
    with pytest.raises(ValueError, match='display'):
        server.call('delete_pair', args)
    shown = server.call('get_item', {'kind':'pair', 'id':pair['id']})
    args['view_token'] = shown['view_token']
    args['expected_sha'] = '0' * 64
    with pytest.raises(ValueError):
        server.call('delete_pair', args)
    args['expected_sha'] = pair['content_sha256']
    assert server.call('delete_pair', args)['deleted'] is True
    with pytest.raises(ValueError):
        server.call('delete_pair', args)


def test_delete_stale_pair_and_matching_copies(app, env):
    j, r, cfg, conn, pair = env
    old = storage.insert_record(conn, 'character_pairs', dict(pair, id='old', constitution_sha256='old'))
    conn.commit()
    app.delete_pair(old['id'], old['content_sha256'])
    assert storage.records(conn, 'character_pairs') == []
    assert app.queue()['counts']['pairs'] == 0


def test_delete_saves_optional_reason_without_keeping_pair_text(server, env):
    j, r, cfg, conn, pair = env
    shown = server.call('get_item', dict(kind='pair', id=pair['id']))
    server.call('delete_pair', dict(id=pair['id'], expected_sha=pair['content_sha256'],
        view_token=shown['view_token'], reason='  I no longer work on this project.  '))
    event = storage.records(conn, 'character_review_events')[0]
    assert event['reason'] == 'I no longer work on this project.'
    assert event['subject']['topic'] == 'Task'
    assert 'Short' not in str(event) and 'Long' not in str(event)
    assert storage.records(conn, 'character_pairs') == []


@pytest.mark.parametrize('reason', [None, 123, 'x' * 2001])
def test_invalid_deletion_reason_leaves_pair_untouched(app, env, reason):
    j, r, cfg, conn, pair = env
    before = storage.snapshot(conn)
    with pytest.raises(ValueError, match='reason'):
        app.delete_pair(pair['id'], pair['content_sha256'], reason=reason)
    assert storage.snapshot(conn) == before


def test_pair_detail_shows_the_real_user_feedback_with_source_hash_check(app, env):
    j, r, cfg, conn, pair = env
    moment = storage.insert_record(conn, 'character_moments', dict(id='moment-feedback', revision=1,
        constitution_sha256='abc', user_correction='Nie twierdź, że działa, zanim sprawdzisz.',
        is_correction=True, context_sufficient=True, agreement='agrees'))
    storage.update_record(conn, 'character_pairs', dict(pair, revision=2, moment_id=moment['id'],
        provenance={**pair['provenance'], 'source_content_sha256':moment['content_sha256']}),
        expected_sha=pair['content_sha256'])
    conn.commit()
    assert app.detail('pair', pair['id'])['user_feedback'] == moment['user_correction']
    storage.update_record(conn, 'character_moments', dict(moment, revision=2, user_correction='Changed evidence'),
        expected_sha=moment['content_sha256'])
    conn.commit()
    assert app.detail('pair', pair['id'])['user_feedback'] is None


@pytest.mark.parametrize('change', [{'prompt_version':'previous-quality-filter'}, {'source_kind':'synthetic'}])
def test_pairs_without_current_real_problem_evidence_leave_active_queue(app, env, change):
    j, r, cfg, conn, pair = env
    updated = storage.update_record(conn, 'character_pairs', dict(pair, revision=2, **change),
        expected_sha=pair['content_sha256'])
    conn.commit()
    queue = app.queue()
    assert queue['pairs'] == []
    assert queue['history'][0]['status'] == 'needs_evidence'
    assert app.detail('pair', pair['id'])['can_accept'] is False
    with pytest.raises(ValueError, match='evidence'):
        app.decide('pair', pair['id'], updated['content_sha256'], 'left', 'Reviewed')


def test_all_removed_source_identities_remain_suppressed(app, env):
    _, _, cfg, db, base = env
    sources = []
    for n in (1, 2):
        source = storage.insert_record(db, 'character_moments', dict(
            id=f'moment-{n}', constitution_sha256='abc',
            identity={'session_id': f'session-{n}'}, correction_seq=n,
            messages=base['prompt_messages']))
        sources.append(source)
    first = storage.update_record(db, 'character_pairs', dict(
        base, revision=2, moment_id=sources[0]['id'],
        provenance={'source_content_sha256': sources[0]['content_sha256']}),
        expected_sha=base['content_sha256'])
    storage.insert_record(db, 'character_pairs', dict(
        base, id='second', moment_id=sources[1]['id'],
        provenance={'source_content_sha256': sources[1]['content_sha256']}))
    storage.insert_record(db, 'character_pairs', dict(
        base, id='second-version', moment_id=sources[1]['id'],
        prompt_messages=[{'role': 'user', 'content': 'Earlier context'}, *base['prompt_messages']],
        provenance={'source_content_sha256': sources[1]['content_sha256']}))
    storage.insert_record(db, 'character_pairs', dict(
        base, id='unrelated', prompt_messages=[{'role': 'user', 'content': 'Unrelated task'}]))
    db.commit()
    app.delete_pair(first['id'], first['content_sha256'])
    assert [row['id'] for row in storage.records(db, 'character_pairs')] == ['unrelated']
    changed_context_for_second_source = dict(
        identity=sources[1]['identity'], correction_seq=2,
        messages=[{'role':'user', 'content':'Extra earlier context'}, *base['prompt_messages']])
    assert storage.is_deleted(db, changed_context_for_second_source)


def test_stale_pair_can_be_deleted_while_new_constitution_is_draft(app, env, monkeypatch):
    from character import constitution
    _, review, cfg, db, pair = env
    cfg.data['character']['constitution_path'] = str(cfg.db_path.parent / 'draft-constitution')
    from pathlib import Path
    Path(cfg.data['character']['constitution_path']).write_text(
        '---\nstatus: draft\nversion: 2\n---\n## Traits\n'
        '### trait-brief\nBe brief.\nWhy: Clear replies.\n'
        '## Priority order\nBrief.\n## Hard constraints\nBe honest.\n')
    monkeypatch.setattr(review, 'load_constitution', constitution.load_constitution)
    monkeypatch.setitem(app.queue.__globals__, 'load_constitution', constitution.load_constitution)
    assert app.queue()['history'][0]['status'] == 'stale'
    assert app.delete_pair(pair['id'], pair['content_sha256'])['deleted']
