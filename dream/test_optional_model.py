"""Tests for the optional per-stage model.

A stage may omit `model` so the provider falls back to its own default model.
Explicit models remain settable via TOML, stage-specific environment variables,
and the CLI `--model` override.
"""
import sqlite3
from pathlib import Path

import pytest

import backend
import codex_cli
import claude_cli
from config import load_config
from distill import sessions_needing_distill
from model_types import GenerationResult


ROOT = Path(__file__).parent
SCHEMA = (ROOT / "schema.sql").read_text(encoding="utf-8")


def _config_without_model(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        "[stages.distill]\nprovider = \"codex\"\n"
        "[stages.consolidate]\nprovider = \"codex\"\n",
        encoding="utf-8",
    )
    return path


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    return conn


def test_stage_without_model_is_valid(tmp_path):
    config = load_config(_config_without_model(tmp_path))
    assert "model" not in config.stage("distill")
    assert "model" not in config.stage("consolidate")


def test_default_config_has_no_hardcoded_model():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    assert "model" not in config.stage("distill")
    assert "model" not in config.stage("consolidate")


def test_explicit_model_settable_via_toml_env_and_override(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        "[stages.distill]\nprovider = \"codex\"\nmodel = \"toml-model\"\n"
        "[stages.consolidate]\nprovider = \"codex\"\n",
        encoding="utf-8",
    )
    assert load_config(path).stage("distill")["model"] == "toml-model"

    monkeypatch.setenv("DREAM_DISTILL_MODEL", "env-model")
    assert load_config(path).stage("distill")["model"] == "env-model"

    monkeypatch.delenv("DREAM_DISTILL_MODEL")
    overridden = load_config(path, overrides={"stages": {"distill": {"model": "cli-model"}}})
    assert overridden.stage("distill")["model"] == "cli-model"


def _capture_generate(monkeypatch):
    captured: dict = {}

    class FakeProvider:
        def generate_structured(self, request):
            captured["request"] = request
            return GenerationResult(
                output={}, raw_result="{}", provider="fake", model=None,
                usage=None, duration_ms=1,
            )

    monkeypatch.setattr(backend, "create_provider", lambda _type: FakeProvider())
    return captured


def test_generate_passes_none_model_when_stage_omits_model(monkeypatch, tmp_path):
    captured = _capture_generate(monkeypatch)
    config = load_config(_config_without_model(tmp_path))
    backend.generate("distill", "prompt", {"type": "object"}, config=config)
    assert captured["request"].model is None


def test_generate_passes_explicit_stage_model(monkeypatch, tmp_path):
    captured = _capture_generate(monkeypatch)
    path = _config_without_model(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "[stages.distill]\nprovider = \"codex\"\n",
            "[stages.distill]\nprovider = \"codex\"\nmodel = \"m-1\"\n",
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    backend.generate("distill", "prompt", {"type": "object"}, config=config)
    assert captured["request"].model == "m-1"


def test_generate_cli_override_wins_over_stage_model(monkeypatch, tmp_path):
    captured = _capture_generate(monkeypatch)
    config = load_config(_config_without_model(tmp_path))
    backend.generate("distill", "prompt", {"type": "object"}, model="override", config=config)
    assert captured["request"].model == "override"


def test_codex_call_without_model_omits_model_flag(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise RuntimeError("stop here")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop here"):
        codex_cli.call_codex("p", {"type": "object"}, model=None)
    cmd = captured["cmd"]
    assert cmd[:2] == ["codex", "exec"]
    assert "-m" not in cmd
    assert "-c" not in cmd  # no reasoning effort supplied either


def test_codex_call_without_model_keeps_reasoning_effort(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise RuntimeError("stop here")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop here"):
        codex_cli.call_codex(
            "p", {"type": "object"}, model=None, reasoning_effort="low",
        )
    cmd = captured["cmd"]
    assert "-m" not in cmd
    assert "-c" in cmd and "model_reasoning_effort=low" in cmd


def test_codex_call_with_explicit_model_still_sends_model(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise RuntimeError("stop here")

    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop here"):
        codex_cli.call_codex("p", {"type": "object"}, model="m-1", reasoning_effort="high")
    cmd = captured["cmd"]
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "m-1"
    assert "model_reasoning_effort=high" in cmd


def test_claude_call_without_model_omits_model_flag(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise RuntimeError("stop here")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop here"):
        claude_cli.call_claude("p", {"type": "object"}, model=None)
    cmd = captured["cmd"]
    assert "--model" not in cmd


def test_claude_call_with_explicit_model_still_sends_model(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        raise RuntimeError("stop here")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop here"):
        claude_cli.call_claude("p", {"type": "object"}, model="m-1")
    cmd = captured["cmd"]
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "m-1"


def test_sessions_needing_distill_without_model_in_stage(tmp_path):
    config = load_config(_config_without_model(tmp_path))
    conn = _db()
    conn.execute(
        "INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime, total_chars) "
        "VALUES ('s1', 'p', '/x', 1.0, 2000)"
    )
    conn.commit()
    assert [sid for sid, _ in sessions_needing_distill(conn, config=config)] == ["s1"]


def test_default_memory_root_follows_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(Path("/nonexistent/dream-config.toml"))
    home_slug = str(tmp_path).replace("/", "-")
    assert config.memory_root == tmp_path / ".claude" / "projects" / home_slug / "memory"
