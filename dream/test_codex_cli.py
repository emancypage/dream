import json
import subprocess
from pathlib import Path

import codex_cli


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
    assert not any("dangerously-bypass" in arg for arg in cmd)
    assert not any(str(memory_root) in arg for arg in cmd)
