"""Hermetic CLI workflow across generation, review, deployment, eval and cleanup."""
import json
from pathlib import Path
from types import SimpleNamespace

import backend
import dream
from character import storage
from test_character_constitution import TEXT


def test_complete_character_cli_workflow_with_fake_providers(tmp_path, monkeypatch, capsys):
    memory = tmp_path / 'memory'; memory.mkdir()
    rules = tmp_path / 'rules.md'; rules.write_text(TEXT)
    original_rules = rules.read_bytes()
    database = tmp_path / 'dream.db'
    state = tmp_path / 'state'
    config = tmp_path / 'config.toml'
    config.write_text(
        '[storage]\ndb_path="' + str(database) + '"\nmemory_root="' + str(memory) + '"\n'
        '[character]\nenabled=true\nconstitution_path="' + str(rules) + '"\nstate_root="' + str(state) + '"\n'
        + ''.join('\n[stages.' + stage + ']\nmodel="fake-model"\n' for stage in ('character_extract', 'character_generate', 'character_judge', 'character_eval_response')))
    conn = dream.open_db(database)
    conn.execute("INSERT INTO sessions(session_id,source,source_revision,parser_version,project_slug,jsonl_path,jsonl_mtime) VALUES('session','codex','r1','v1','private-project','/fixture',0)")
    messages = [('user', 'What tests prove it?'), ('assistant', 'Trust me.'), ('user', 'Actually use deterministic evidence.')]
    conn.executemany('INSERT INTO messages(session_id,seq,role,text) VALUES(?,?,?,?)', [('session', i, role, text) for i, (role, text) in enumerate(messages)])
    conn.commit(); conn.close()
    calls = []
    def fake(stage, prompt, schema, **kwargs):
        calls.append(stage)
        if stage == 'character_extract':
            output = {'is_correction': True, 'context_sufficient': True, 'context_excerpt': messages[0][1], 'corrected_assistant_response': messages[1][1], 'user_correction': messages[2][1], 'trait_ids': ['trait-01'], 'agreement': 'agrees', 'rationale': 'The original answer supplied no test evidence.'}
        elif stage in ('character_generate', 'character_eval_response'):
            output = {'response': 'Run the test and report its output.'}
        elif 'scores' in schema['properties']:
            dimensions = schema['properties']['scores']['properties']['A']['properties']
            output = {'scores': {label: {d: 3 for d in dimensions} for label in ('A', 'B')}, 'rationale': 'Both answers follow the supplied rubric.'}
        else:
            payload = json.loads(prompt)
            preferred = 'A' if payload['A'].startswith('Run') else 'B'
            output = {'winner': preferred, 'assessments': [{'trait_id': 'trait-01', 'a_assessment': 'supports' if preferred == 'A' else 'violates', 'b_assessment': 'supports' if preferred == 'B' else 'violates', 'rationale': 'The response asks for deterministic evidence.'}], 'rationale': 'Prefer a checkable answer.'}
        return SimpleNamespace(output=output, provider='fake', model='fake-model', usage=None, duration_ms=1)
    monkeypatch.setattr(backend, 'generate', fake)
    def command(*args):
        assert dream.main(['--config', str(config), *args]) == 0
        return json.loads(capsys.readouterr().out)
    assert command('character', 'estimate', '--limit', '1')['source_items'] == 1
    assert calls == []
    assert command('character', 'generate', '--limit', '1')['generated'] == 1
    assert command('character', 'judge', '--limit', '1')['judged'] == 1
    pair = command('character', 'list')[0]
    command('character', 'accept', pair['id'], '--expected-sha', pair['content_sha256'], '--reason', 'I reviewed the exact response and its evidence.')
    accepted = command('character', 'show', pair['id'])
    body = tmp_path / 'short.json'
    body.write_text(json.dumps(dict(situation='The user asks for proof.', preferred='Run the test and report its output.', avoid='Trust me.', feedback='Report test evidence.', limits='State what the test covers.')))
    draft = command('character', 'short', 'draft', pair['id'], '--expected-pair-sha', accepted['content_sha256'], '--body-file', str(body))
    command('character', 'short', 'approve', draft['id'], '--expected-sha', draft['content_sha256'], '--reason', 'Reviewed the exact short wording.')
    command('character', 'render')
    command('character', 'export', '--format', 'dpo-jsonl')
    exported = json.loads((state / 'accepted.dpo.jsonl').read_text())
    assert exported['metadata']['format'] == 'dream-dpo-v1'
    assert exported['chosen'].startswith('Run')
    fixture = Path(__file__).parent / 'fixtures/character/public.json'
    evaluation = command('character', 'eval', '--fixtures', str(fixture), '--limit', '5')
    report = command('character', 'eval', '--report', evaluation['run_id'])
    assert any(row['valid'] for row in report['categories'])
    assert report['manifest']['result_ids']
    assert list(state.glob('eval-*.md'))
    policy = memory / '.dream-retention.json'
    policy.write_text(json.dumps({'project_slugs': ['private-project']}))
    policy_bytes = policy.read_bytes()
    cleanup = command('retention', '--apply')
    assert not cleanup['character']['rerun_required']
    conn = dream.open_db(database)
    assert storage.records(conn, 'character_pairs') == []
    assert storage.records(conn, 'character_eval_results') == []
    assert conn.execute('SELECT count(*) FROM messages').fetchone()[0] == 3
    conn.close()
    assert not (state / 'block.md').exists()
    assert not (state / 'accepted.dpo.jsonl').exists()
    assert rules.read_bytes() == original_rules
    assert policy.read_bytes() == policy_bytes
