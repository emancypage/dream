import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

import backend
import codex_cli
import dream
import metrics
import stage_runner
from config import load_config
from distill import DISTILL_SCHEMA
from model_types import ProviderError


NOTES = {
    "summary": "The user chose a short daily report.",
    "user_facts": [], "preferences": ["The user wants short daily reports."],
    "projects": [], "references": [], "decisions": [], "open_threads": [],
}


def _codex_process(monkeypatch, tmp_path, outcomes):
    calls, delays = [], []
    outcomes = iter(outcomes)

    def run(cmd, **kwargs):
        output_path = Path(cmd[cmd.index("-o") + 1])
        calls.append((list(cmd), kwargs, output_path.exists()))
        kind, value = next(outcomes)
        if kind == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        if kind in ("error", "signal"):
            output_path.write_text('{"stale":true}')
            event = {"type": "turn.failed", "error": {"message": value}}
            return subprocess.CompletedProcess(cmd, -15 if kind == "signal" else 1,
                                               stdout=json.dumps(event), stderr="startup warning")
        if kind == "ok":
            output_path.write_text(json.dumps(value))
        elif kind == "invalid-json":
            output_path.write_text(value)
        return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')

    monkeypatch.setattr(codex_cli, "ensure_workdir", lambda: tmp_path)
    monkeypatch.setattr(codex_cli.subprocess, "run", run)
    monkeypatch.setattr(codex_cli.time, "sleep", delays.append)
    return calls, delays


def _config(tmp_path):
    return load_config(tmp_path / "absent.toml", overrides={
        "storage": {"memory_root": str(tmp_path / "memory")},
        "stages": {"distill": {"provider": "codex", "model": "test-model", "timeout_seconds": 60}},
    })


@pytest.mark.parametrize("message", [
    "stream disconnected before completion: connection reset by peer",
    "error sending request for url (https://example.invalid/responses)",
    "request timed out",
    "unexpected status 500 Internal Server Error",
    "unexpected status 502 Bad Gateway",
    "unexpected status 503 Service Unavailable",
    "unexpected status 504 Gateway Timeout",
    "HTTP 503 Service Unavailable",
    "Internal server error",
    "rate_limit_exceeded: please slow down",
    "Rate limit exceeded, please retry later",
    "Selected model is at capacity. Please try a different model.",
])
def test_summary_retries_one_temporary_codex_error(monkeypatch, tmp_path, message, capsys):
    calls, delays = _codex_process(monkeypatch, tmp_path, [("error", message), ("ok", NOTES)])
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-be-used")

    result = backend.generate("distill", "test transcript", DISTILL_SCHEMA, config=_config(tmp_path))

    assert result.output == NOTES
    assert len(calls) == 2
    assert delays == [5]
    assert calls[0][:2] == calls[1][:2]
    assert not calls[1][2], "A failed attempt's output must be cleared before retrying"
    assert calls[1][1]["input"].endswith("test transcript")
    assert calls[1][1]["timeout"] == 60
    assert "OPENAI_API_KEY" not in calls[1][1]["env"]
    assert "CODEX_API_KEY" not in calls[1][1]["env"]
    assert "retrying once" in capsys.readouterr().err


def test_second_temporary_failure_stops_and_reports_the_final_error(monkeypatch, tmp_path):
    calls, delays = _codex_process(monkeypatch, tmp_path, [
        ("error", "request timed out"),
        ("error", "unexpected status 503 Service Unavailable"),
        ("ok", NOTES),
    ])

    with pytest.raises(ProviderError, match="503 Service Unavailable"):
        backend.generate("distill", "test transcript", DISTILL_SCHEMA, config=_config(tmp_path))

    assert len(calls) == 2
    assert delays == [5]


@pytest.mark.parametrize("message", [
    "unexpected status 401 Unauthorized",
    "unexpected status 403 Forbidden",
    "unexpected status 400 Bad Request",
    "stream disconnected before completion: authentication token expired",
    "stream disconnected before completion: usage limit reached",
    "stream disconnected before completion: insufficient_quota",
    "stream disconnected before completion: invalid schema",
    "stream disconnected before completion: context_length_exceeded",
    "stream disconnected before completion: unexpected status 401 Unauthorized",
    "error sending request for url: invalid peer certificate",
    "unexpected status 401 Unauthorized " + "x" * 2200 + " stream disconnected before completion",
    "unexpected status 429: usage_limit_reached",
    "stream disconnected before completion: The requested model 'test-model' is not supported",
    "stream disconnected before completion: Your access token has expired. Please sign in again.",
    "stream disconnected before completion: insufficient_credit_balance",
    "The model refused the request",
    "Unknown Codex failure",
])
def test_summary_does_not_retry_permanent_or_unknown_errors(monkeypatch, tmp_path, message):
    calls, delays = _codex_process(monkeypatch, tmp_path, [("error", message), ("ok", NOTES)])

    with pytest.raises(ProviderError):
        backend.generate("distill", "test transcript", DISTILL_SCHEMA, config=_config(tmp_path))

    assert len(calls) == 1
    assert delays == []


@pytest.mark.parametrize("stage", ["consolidate", "review", "character_extract", "character_generate"])
def test_other_stages_do_not_retry_codex_errors(monkeypatch, tmp_path, stage):
    calls, delays = _codex_process(monkeypatch, tmp_path, [("error", "request timed out"), ("ok", NOTES)])

    with pytest.raises(ProviderError):
        backend.generate(stage, "test transcript", DISTILL_SCHEMA, model="test-model", config=_config(tmp_path))

    assert len(calls) == 1
    assert delays == []


def test_summary_does_not_enable_retry_for_other_providers(monkeypatch, tmp_path):
    calls = []

    class OtherProvider:
        def generate_structured(self, request):
            calls.append(request)
            raise ProviderError("request timed out")

    monkeypatch.setattr(backend, "create_provider", lambda *args: OtherProvider())
    config = load_config(tmp_path / "absent.toml", overrides={
        "stages": {"distill": {"provider": "claude"}},
    })
    with pytest.raises(ProviderError, match="request timed out"):
        backend.generate("distill", "test transcript", DISTILL_SCHEMA, config=config)

    assert len(calls) == 1
    assert not calls[0].options.get("retry_temporary_errors")


@pytest.mark.parametrize("outcome", [
    ("ok", {"summary": "missing required fields"}),
    ("invalid-json", "not JSON"),
    ("no-output", None),
    ("timeout", None),
    ("signal", "stream disconnected before completion"),
])
def test_summary_does_not_retry_invalid_output_or_interrupted_process(monkeypatch, tmp_path, outcome):
    calls, delays = _codex_process(monkeypatch, tmp_path, [outcome, ("ok", NOTES)])

    with pytest.raises(ProviderError):
        backend.generate("distill", "test transcript", DISTILL_SCHEMA, config=_config(tmp_path))

    assert len(calls) == 1
    assert delays == []


@pytest.mark.parametrize("recover", [True, False])
def test_summary_retry_preserves_database_and_stage_outcome(monkeypatch, tmp_path, recover):
    conn = sqlite3.connect(":memory:")
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())
    conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime,total_chars) "
                 "VALUES ('test-session','test-project','test.jsonl',1,600)")
    conn.execute("INSERT INTO messages(session_id,seq,role,text) VALUES ('test-session',0,'user',?)",
                 ("Please keep daily reports short. " * 20,))
    conn.commit()
    monkeypatch.setattr(dream, "open_db", lambda path: conn)
    outcomes = [("error", "request timed out"), ("ok", NOTES) if recover else ("error", "request timed out")]
    calls, delays = _codex_process(monkeypatch, tmp_path, outcomes)
    state_path = tmp_path / "run.json"
    monkeypatch.setattr(stage_runner, "STATE_PATH", state_path)
    metrics.start_run(state_path)
    args = argparse.Namespace(db=tmp_path / "dream.db", config=_config(tmp_path), min_chars=500,
                              project=None, model=None, refresh=False, limit=50, yes=True, max_sessions=5)

    status = stage_runner.run_stage("distill", [], runner=lambda command: dream.cmd_distill(args),
                                    continue_on_error=True)
    state = metrics.finish_run(state_path, success=True)

    assert status == 0
    assert len(calls) == 2
    assert delays == [5]
    assert conn.execute("SELECT COUNT(*) FROM distilled").fetchone()[0] == (1 if recover else 0)
    assert state["status"] == ("success" if recover else "failed")
    if not recover:
        assert state["failed_stages"] == [{"stage": "distill", "error_code": "stage_exit_1"}]
