import os

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
