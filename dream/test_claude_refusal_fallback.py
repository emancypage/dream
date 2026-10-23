"""Subscription-backed retries for structured Claude refusals."""

import json
from types import SimpleNamespace

import pytest

import backend
import claude_cli
from config import ConfigError, load_config
from model_types import ProviderError, SchemaValidationError


SCHEMA = {
    'type': 'object',
    'properties': {'value': {'type': 'string'}},
    'required': ['value'],
    'additionalProperties': False,
}
SUCCESS = {
    'is_error': False,
    'structured_output': {'value': 'done'},
    'usage': {'input_tokens': 5, 'output_tokens': 3},
    'total_cost_usd': 0.2,
}
REFUSAL = {
    'is_error': True,
    'stop_reason': 'refusal',
    'result': 'The request was flagged.',
    'usage': {'input_tokens': 2, 'output_tokens': 0},
    'total_cost_usd': 0.1,
}


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_cli, 'WORKDIR', tmp_path / 'workdir')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key-must-not-be-used')
    calls = []

    def replies(*responses):
        pending = iter(responses)

        def run(cmd, **kwargs):
            calls.append((list(cmd), kwargs))
            code, output = next(pending)
            return SimpleNamespace(returncode=code, stdout=json.dumps(output), stderr='')

        monkeypatch.setattr(claude_cli.subprocess, 'run', run)

    return calls, replies


@pytest.mark.parametrize('returncode', [0, 1])
def test_configured_refusal_retries_same_request_and_records_fallback(worker, tmp_path, returncode):
    calls, replies = worker
    replies((returncode, REFUSAL), (0, SUCCESS))
    cfg = load_config(tmp_path / 'missing', {
        'stages': {'review': {'provider': 'claude', 'model': 'claude-opus-5-5'}},
        'providers': {'claude': {'refusal_fallback_model': 'claude-opus-4-8'}},
    })

    result = backend.generate('review', 'same input', SCHEMA, config=cfg, system_prompt='same rules')

    assert result.output == {'value': 'done'}
    assert result.model == 'claude-opus-4-8'
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 3
    assert result.total_cost_usd == pytest.approx(0.3)
    assert [cmd[cmd.index('--model') + 1] for cmd, _ in calls] == ['claude-opus-5-5', 'claude-opus-4-8']
    for cmd, options in calls:
        assert options['input'] == 'same input'
        assert cmd[cmd.index('--system-prompt') + 1] == 'same rules'
        assert json.loads(cmd[cmd.index('--json-schema') + 1]) == SCHEMA
        assert 'ANTHROPIC_API_KEY' not in options['env']


def test_primary_success_does_not_call_fallback(worker):
    calls, replies = worker
    replies((0, SUCCESS))
    result = claude_cli.call_claude('input', SCHEMA, model='claude-opus-5-5', refusal_fallback_model='claude-opus-4-8')
    assert result.model == 'claude-opus-5-5'
    assert result.total_cost_usd == 0.2
    assert len(calls) == 1


@pytest.mark.parametrize('fallback', [None, 'claude-opus-5-5'])
def test_refusal_without_a_distinct_fallback_is_not_retried(worker, fallback):
    calls, replies = worker
    replies((1, REFUSAL))
    with pytest.raises(ProviderError):
        claude_cli.call_claude('input', SCHEMA, model='claude-opus-5-5', refusal_fallback_model=fallback)
    assert len(calls) == 1


def test_second_refusal_is_not_retried_or_accepted_as_partial_output(worker):
    calls, replies = worker
    replies((1, REFUSAL), (0, dict(REFUSAL, structured_output={'value': 'partial'})))
    with pytest.raises(ProviderError, match='refusal'):
        claude_cli.call_claude('input', SCHEMA, model='claude-opus-5-5', refusal_fallback_model='claude-opus-4-8')
    assert len(calls) == 2


def test_other_errors_do_not_trigger_fallback(worker):
    calls, replies = worker
    replies((1, {'is_error': True, 'result': 'Unavailable, not a refusal', 'stop_reason': 'end_turn'}))
    with pytest.raises(ProviderError, match='exited 1'):
        claude_cli.call_claude('input', SCHEMA, model='claude-opus-5-5', refusal_fallback_model='claude-opus-4-8')
    assert len(calls) == 1


def test_fallback_output_still_must_match_schema(worker):
    _, replies = worker
    replies((1, REFUSAL), (0, dict(SUCCESS, structured_output={'value': 7})))
    with pytest.raises(SchemaValidationError):
        claude_cli.call_claude('input', SCHEMA, model='claude-opus-5-5', refusal_fallback_model='claude-opus-4-8')


def test_retry_uses_only_the_remaining_timeout(worker, monkeypatch):
    calls, replies = worker
    replies((1, REFUSAL), (0, SUCCESS))
    clock = {'now': 10.0}
    monkeypatch.setattr(claude_cli.time, 'monotonic', lambda: clock['now'])
    run = claude_cli.subprocess.run

    def timed_run(*args, **kwargs):
        result = run(*args, **kwargs)
        clock['now'] += 2 if len(calls) == 1 else 3
        return result

    monkeypatch.setattr(claude_cli.subprocess, 'run', timed_run)
    result = claude_cli.call_claude('input', SCHEMA, model='claude-opus-5-5', refusal_fallback_model='claude-opus-4-8', timeout=5)
    assert [options['timeout'] for _, options in calls] == [5, 3]
    assert result.duration_ms == 5000


@pytest.mark.parametrize('provider,value', [('claude', ''), ('claude', 7), ('codex', 'claude-opus-4-8')])
def test_invalid_refusal_fallback_configuration_is_rejected(tmp_path, provider, value):
    with pytest.raises(ConfigError, match='refusal_fallback_model'):
        load_config(tmp_path / 'missing', {'providers': {provider: {'refusal_fallback_model': value}}})
