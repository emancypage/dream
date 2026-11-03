"""Human form answers must survive transcript ingestion without importing tool noise."""
import json
import sqlite3
from pathlib import Path

import pytest

from ingest import ingest_parsed_session
from sources.claude_jsonl import ClaudeJSONLSource


def form_rows(answer):
    return [
        {'type': 'user', 'message': {'content': 'Consult the migration plan.'}},
        {'type': 'assistant', 'message': {'content': [
            {'type': 'text', 'text': 'Please clarify these decisions.'},
            {'type': 'tool_use', 'id': 'form-1', 'name': 'AskUserQuestion', 'input': {
                'questions': [{'question': 'Where should logs go?', 'header': 'Logs',
                               'multiSelect': False, 'options': [
                                   {'label': 'Existing Grafana', 'description': 'Reuse it.'},
                                   {'label': 'Remove logs', 'description': 'No monitoring.'}]}]}},
            {'type': 'tool_use', 'id': 'shell-1', 'name': 'Bash',
             'input': {'command': 'private command'}},
        ]}},
        {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'shell-1', 'content': 'private tool output'},
            {'type': 'tool_result', 'tool_use_id': 'form-1', 'content': answer},
            {'type': 'text', 'text': 'Keep the analyzer too.'},
        ]}},
        {'type': 'assistant', 'message': {'content': [
            {'type': 'text', 'text': 'I will reuse Grafana and keep the analyzer.'}]}},
    ]


def parse_rows(tmp_path, rows):
    root = tmp_path / 'history'
    project = root / 'project'
    project.mkdir(parents=True)
    path = project / 'session.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    source = ClaudeJSONLSource(root)
    return source.parse(next(iter(source.discover())))


@pytest.mark.parametrize('answer', [
    'User has answered your questions: "Where should logs go?"="Existing Grafana"',
    [{'type': 'text', 'text': 'Existing Grafana; commit on a branch.'}],
])
def test_form_questions_and_answers_reach_stored_transcript(tmp_path, answer):
    parsed = parse_rows(tmp_path, form_rows(answer))
    conn = sqlite3.connect(':memory:')
    conn.executescript((Path(__file__).parent / 'schema.sql').read_text())
    ingest_parsed_session(parsed, conn)
    turns = conn.execute('SELECT role,text FROM messages ORDER BY seq').fetchall()
    assert [role for role, _ in turns] == ['user', 'assistant', 'user', 'assistant']
    assert 'Where should logs go?' in turns[1][1]
    assert 'Remove logs' in turns[1][1]
    assert 'Existing Grafana' in turns[2][1]
    assert 'Keep the analyzer too.' in turns[2][1]
    assert 'private command' not in str(turns)
    assert 'private tool output' not in str(turns)


def test_unrelated_and_sidechain_form_results_are_not_human_answers(tmp_path):
    rows = form_rows('Existing Grafana')
    rows.insert(0, {'type': 'assistant', 'isSidechain': True, 'message': {'content': [
        {'type': 'tool_use', 'id': 'side-form', 'name': 'AskUserQuestion',
         'input': {'questions': [{'question': 'Sidechain question'}]}}]}})
    rows.append({'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 'side-form', 'content': 'sidechain answer'},
        {'type': 'tool_result', 'tool_use_id': 'unknown', 'content': 'unmatched answer'}]}})
    text = str(parse_rows(tmp_path, rows).messages)
    assert 'sidechain answer' not in text
    assert 'unmatched answer' not in text


def test_parser_upgrade_reimports_unchanged_transcript(tmp_path):
    from dataclasses import replace
    from sources.base import Message

    parsed = parse_rows(tmp_path, form_rows('Existing Grafana'))
    conn = sqlite3.connect(':memory:')
    conn.executescript((Path(__file__).parent / 'schema.sql').read_text())
    old = replace(parsed, parser_version='claude-jsonl-v2', messages=[
        Message(1, 'user', None, 'Consult the migration plan.')])
    ingest_parsed_session(old, conn)
    changed, _ = ingest_parsed_session(parsed, conn)
    assert changed
    assert 'Existing Grafana' in str(conn.execute('SELECT text FROM messages').fetchall())
