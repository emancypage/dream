import os

import pytest

from model_types import child_env
from providers.registry import preflight_provider


def test_child_env_prepends_directory_of_absolute_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    env = child_env(str(tmp_path / "bin" / "codex"))
    assert env["PATH"].split(os.pathsep) == [str(tmp_path / "bin"), "/usr/bin"]


def test_child_env_keeps_path_for_bare_name(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    assert child_env("codex")["PATH"] == "/usr/bin"


def test_preflight_fails_when_executable_cannot_start(tmp_path):
    exe = tmp_path / "codex"
    exe.write_text("#!/usr/bin/env dream-missing-interpreter\n")
    exe.chmod(0o755)
    ok, detail = preflight_provider({"type": "codex-cli", "executable": str(exe)})
    assert ok is False and "cannot start" in detail


def test_preflight_reports_version_using_interpreter_beside_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    fake = tmp_path / "fakenode"
    fake.write_text("#!/bin/sh\necho 'tool 1.2.3'\n")
    fake.chmod(0o755)
    exe = tmp_path / "codex"
    exe.write_text("#!/usr/bin/env fakenode\n")
    exe.chmod(0o755)
    assert preflight_provider({"type": "codex-cli", "executable": str(exe)}) == (True, f"{exe} (tool 1.2.3)")


@pytest.mark.parametrize("cli_first", [False, True])
def test_bare_preflight_uses_the_same_interpreter_path_as_execution(tmp_path, monkeypatch, cli_first):
    cli_dir, runtime_dir = tmp_path / "cli", tmp_path / "runtime"
    cli_dir.mkdir()
    runtime_dir.mkdir()
    for directory, version in [(cli_dir, "cli-runtime"), (runtime_dir, "other-runtime")]:
        runtime = directory / "dream-test-node"
        runtime.write_text(f"#!/bin/sh\necho {version}\n")
        runtime.chmod(0o755)
    exe = cli_dir / "codex"
    exe.write_text("#!/usr/bin/env dream-test-node\n")
    exe.chmod(0o755)
    order = [cli_dir, runtime_dir] if cli_first else [runtime_dir, cli_dir]
    monkeypatch.setenv("PATH", os.pathsep.join(map(str, order)))
    expected = "cli-runtime" if cli_first else "other-runtime"
    assert preflight_provider({"type": "codex-cli", "executable": "codex"}) == (True, f"{exe} ({expected})")


@pytest.mark.parametrize("hooks", [False, True])
@pytest.mark.parametrize("active_codex", [False, True])
def test_public_preflight_only_requires_active_provider_executables(tmp_path, monkeypatch, capsys, hooks, active_codex):
    import dream

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("DREAM_CONTROL_STATE_DIR", str(tmp_path / "control"))
    for key in ("CODEX_HOME", "OPENCODE_CONFIG", "XDG_CONFIG_HOME", "DREAM_MEMORY_ROOT", "CLAUDE_DREAM_DB"):
        monkeypatch.delenv(key, raising=False)
    cli = tmp_path / "external" / "codex"
    cli.parent.mkdir()
    cli.write_text("#!/usr/bin/env dream-test-node\n")
    runtime = cli.parent / "dream-test-node"
    runtime.write_text("#!/bin/sh\necho 'configured-cli 1.0'\n")
    runtime.chmod(0o755)
    cli.chmod(0o755)
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "auth.json").write_text("{}")
    claude = tmp_path / "fake-claude"
    claude.write_text("#!/bin/sh\necho alternate-cli\n")
    claude.chmod(0o755)
    config_path, db_path = tmp_path / "config.toml", tmp_path / "dream.db"
    dream.open_db(db_path).close()
    config_path.write_text(
        f"[recall]\ninstall_hooks={str(hooks).lower()}\n"
        f"[providers.codex]\nexecutable='{cli if active_codex else tmp_path / 'missing-codex'}'\n"
        f"[providers.claude]\nexecutable='{claude}'\n"
        + ("" if active_codex else "[stages.distill]\nprovider='claude'\n[stages.consolidate]\nprovider='claude'\n[stages.review]\nprovider='claude'\n[stages.audit]\nprovider='claude'\n")
    )
    assert dream.main(["--config", str(config_path), "--db", str(db_path), "preflight"]) == 0
    output = capsys.readouterr().out
    assert "FAIL" not in output
    assert "codex.version" not in output
    assert ("configured-cli 1.0" in output) == active_codex
