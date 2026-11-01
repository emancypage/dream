import json
import subprocess
from pathlib import Path

import codex_cli
import pytest
from model_types import ProviderError


def test_codex_command_uses_isolated_workspace_write_sandbox(monkeypatch, tmp_path):
    workdir = tmp_path / "codex-workdir"
    memory_root = tmp_path / "memory"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(codex_cli, "ensure_workdir", lambda: workdir)
    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    codex_cli.call_codex(
        "return the JSON object",
        {
            "type": "object",
            "required": ["ok"],
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
        },
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("-s") + 1] == "workspace-write"
    assert cmd[cmd.index("-C") + 1] == str(workdir)
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--json" in cmd
    assert not any("dangerously-bypass" in arg for arg in cmd)
    assert not any(str(memory_root) in arg for arg in cmd)


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("event", [
    {"type": "turn.failed", "error": {"message": "stream disconnected before completion"}},
    {"type": "error", "message": "stream disconnected before completion"},
])
def test_codex_reports_event_error_instead_of_startup_warning(monkeypatch, tmp_path, returncode, event):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, returncode,
            stdout="\n".join(json.dumps(value) for value in [
                {"type": "item.completed", "item": {"text": "private transcript text"}},
                event,
            ]),
            stderr="failed to load unrelated skill\n" + "startup information\n" * 100,
        )

    monkeypatch.setattr(codex_cli, "ensure_workdir", lambda: tmp_path)
    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    with pytest.raises(ProviderError) as error:
        codex_cli.call_codex("private transcript text", {"type": "object"})

    assert "stream disconnected before completion" in str(error.value)
    assert "failed to load unrelated skill" not in str(error.value)
    assert "private transcript text" not in str(error.value)


def test_codex_reports_bounded_stderr_tail_without_error_event(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 1,
            stdout='null\n[]\n{"type":"turn.failed","error":null}\ninvalid JSON',
            stderr="startup information\n" * 1000 + "ERROR: authentication token expired",
        )

    monkeypatch.setattr(codex_cli, "ensure_workdir", lambda: tmp_path)
    monkeypatch.setattr(codex_cli.subprocess, "run", fake_run)

    with pytest.raises(ProviderError) as error:
        codex_cli.call_codex("return JSON", {"type": "object"})

    assert "ERROR: authentication token expired" in str(error.value)
    assert len(str(error.value)) < 2200
