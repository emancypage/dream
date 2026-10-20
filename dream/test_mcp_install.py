"""``dream mcp install|uninstall``: idempotent, backed-up client entries.

Every test patches ``HOME`` (and clears ``CODEX_HOME``, ``XDG_CONFIG_HOME`` and
``OPENCODE_CONFIG``) to a temporary directory, so no real client configuration
is read or written. The Claude Code client is driven through a stubbed
``subprocess.run``; no real ``claude`` process starts.
"""

import json
import os
import stat
import tomllib
from pathlib import Path

import pytest

import mcp_install
from mcp_install import CODEX_BEGIN, CODEX_END


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("CODEX_HOME", "XDG_CONFIG_HOME", "OPENCODE_CONFIG", "DREAM_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    assert Path.home() == home
    return home


def _main(capsys, tmp_path, *argv):
    from dream import main
    rc = main(["--config", str(tmp_path / "no-config.toml"), "mcp", *argv])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _backups(path):
    return sorted(path.parent.glob(path.name + ".bak-*"))


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


DREAM_ENTRY = {"command": "dream", "args": ["mcp", "serve"]}


# --- Codex --------------------------------------------------------------------------------


_CODEX_CONFIG = """\
model = "gpt-5"
approval_policy = "on-request"

[features]
memories = false

[mcp_servers.docs]
command = "docs-server"
args = ["--stdio"]
env = { DOCS_TOKEN_ENV = "DOCS_TOKEN" }

[profiles.fast]
model = "gpt-5-mini"
"""


def test_codex_install_creates_the_marker_block_in_a_new_config(home, tmp_path, capsys):
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "codex")
    path = home / ".codex" / "config.toml"
    assert (rc, err) == (0, "")
    text = path.read_text(encoding="utf-8")
    assert text.startswith(CODEX_BEGIN + "\n") and text.endswith(CODEX_END + "\n")
    assert tomllib.loads(text) == {"mcp_servers": {"dream": DREAM_ENTRY}}
    assert _backups(path) == []
    assert str(path) in out and "Restart Codex" in out and "/mcp" in out
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_codex_install_preserves_unrelated_entries_and_round_trips(home, tmp_path, capsys):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text(_CODEX_CONFIG, encoding="utf-8")
    path.chmod(0o640)
    original = path.read_bytes()

    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 0
    installed = path.read_text(encoding="utf-8")
    assert installed.startswith(_CODEX_CONFIG)
    parsed = tomllib.loads(installed)
    assert parsed["mcp_servers"]["dream"] == DREAM_ENTRY
    expected = tomllib.loads(_CODEX_CONFIG)
    expected["mcp_servers"]["dream"] = DREAM_ENTRY
    assert parsed == expected
    backups = _backups(path)
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert str(backups[0]) in out
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o640

    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 0 and "already installed" in out
    assert path.read_text(encoding="utf-8") == installed
    assert _backups(path) == backups

    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "codex")
    assert rc == 0 and "Removed" in out
    assert path.read_bytes() == original
    assert len(_backups(path)) == 2

    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "codex")
    assert rc == 0 and "not installed" in out
    assert path.read_bytes() == original and len(_backups(path)) == 2


def test_codex_install_replaces_a_stale_marker_block_in_place(home, tmp_path, capsys):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    stale = f'{CODEX_BEGIN}\n[mcp_servers.dream]\ncommand = "/old/dream"\nargs = ["mcp", "serve", "--old"]\n{CODEX_END}\n'
    path.write_text('model = "gpt-5"\n\n' + stale + '\n[profiles.fast]\nmodel = "gpt-5-mini"\n', encoding="utf-8")
    rc, _out, _err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 0
    text = path.read_text(encoding="utf-8")
    assert text.count(CODEX_BEGIN) == 1 and "/old/dream" not in text
    assert tomllib.loads(text) == {"model": "gpt-5", "mcp_servers": {"dream": DREAM_ENTRY},
                                   "profiles": {"fast": {"model": "gpt-5-mini"}}}
    assert text.index(CODEX_END) < text.index("[profiles.fast]")


def test_codex_install_appends_after_a_file_without_a_trailing_newline(home, tmp_path, capsys):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text('model = "gpt-5"', encoding="utf-8")
    assert _main(capsys, tmp_path, "install", "--client", "codex")[0] == 0
    assert tomllib.loads(path.read_text(encoding="utf-8")) == {"model": "gpt-5", "mcp_servers": {"dream": DREAM_ENTRY}}
    assert _main(capsys, tmp_path, "uninstall", "--client", "codex")[0] == 0
    assert tomllib.loads(path.read_text(encoding="utf-8")) == {"model": "gpt-5"}


@pytest.mark.parametrize("content, reason", [
    ("model = \n", "not valid TOML"),
    ('[mcp_servers.dream]\ncommand = "someone-else"\n', "not written by dream"),
    ('mcp_servers = { docs = { command = "docs" } }\n', "not valid TOML"),
    (f"{CODEX_BEGIN}\n[mcp_servers.dream]\ncommand = \"dream\"\n", "marker"),
    (f"{CODEX_END}\n{CODEX_BEGIN}\n", "marker"),
    (f"{CODEX_BEGIN}\n{CODEX_END}\n{CODEX_BEGIN}\n{CODEX_END}\n", "marker"),
    ('[profiles.fast]\nmodel = "x"\n' + f'{CODEX_BEGIN}\n[mcp_servers.dream]\ncommand = "dream"\nargs = ["mcp", "serve"]\n'
     f'{CODEX_END}\nextra = 1\n', "unrelated"),
])
def test_codex_install_refuses_configs_it_cannot_edit_safely(home, tmp_path, capsys, content, reason):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")
    before = _files(home)
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 1 and out == ""
    assert err.startswith("dream mcp: ") and reason in err
    assert _files(home) == before


def test_codex_uninstall_refuses_malformed_toml_and_leaves_foreign_entries(home, tmp_path, capsys):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text("model = \n", encoding="utf-8")
    rc, _out, err = _main(capsys, tmp_path, "uninstall", "--client", "codex")
    assert rc == 1 and "not valid TOML" in err
    foreign = '[mcp_servers.dream]\ncommand = "someone-else"\n'
    path.write_text(foreign, encoding="utf-8")
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "codex")
    assert rc == 0 and "not written by dream" in out
    assert path.read_text(encoding="utf-8") == foreign and _backups(path) == []


def test_codex_honours_codex_home_and_an_explicit_path(home, tmp_path, capsys, monkeypatch):
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    assert _main(capsys, tmp_path, "install", "--client", "codex")[0] == 0
    assert tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))["mcp_servers"]["dream"] == DREAM_ENTRY
    explicit = tmp_path / "elsewhere" / "codex.toml"
    assert _main(capsys, tmp_path, "install", "--client", "codex", "--path", str(explicit))[0] == 0
    assert tomllib.loads(explicit.read_text(encoding="utf-8"))["mcp_servers"]["dream"] == DREAM_ENTRY
    assert not (home / ".codex").exists()


def test_codex_install_writes_through_a_symlinked_config(home, tmp_path, capsys):
    real = tmp_path / "dotfiles" / "codex.toml"
    real.parent.mkdir()
    real.write_text('model = "gpt-5"\n', encoding="utf-8")
    link = home / ".codex" / "config.toml"
    link.parent.mkdir()
    link.symlink_to(real)
    assert _main(capsys, tmp_path, "install", "--client", "codex")[0] == 0
    assert link.is_symlink() and link.resolve() == real
    assert tomllib.loads(real.read_text(encoding="utf-8"))["mcp_servers"]["dream"] == DREAM_ENTRY


# --- OpenCode ---------------------------------------------------------------------------------


_OPENCODE = {
    "$schema": "https://opencode.ai/config.json",
    "theme": "tokyonight",
    "provider": {"ollama": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://127.0.0.1:11434/v1"}}},
    "mcp": {"docs": {"type": "remote", "url": "https://docs.example/mcp", "enabled": False}},
}
OPENCODE_ENTRY = {"type": "local", "command": ["dream", "mcp", "serve"], "enabled": True}


def test_opencode_install_preserves_unrelated_entries_and_round_trips(home, tmp_path, capsys):
    path = home / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_OPENCODE, indent=2) + "\n", encoding="utf-8")
    original = path.read_bytes()

    rc, out, err = _main(capsys, tmp_path, "install", "--client", "opencode")
    assert (rc, err) == (0, "")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {**_OPENCODE, "mcp": {**_OPENCODE["mcp"], "dream": OPENCODE_ENTRY}}
    assert list(document) == list(_OPENCODE)
    backups = _backups(path)
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert "Restart OpenCode" in out and str(path) in out

    installed = path.read_bytes()
    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "opencode")
    assert rc == 0 and "already installed" in out
    assert path.read_bytes() == installed and _backups(path) == backups

    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "opencode")
    assert rc == 0 and "Removed" in out
    assert json.loads(path.read_text(encoding="utf-8")) == _OPENCODE
    assert len(_backups(path)) == 2
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "opencode")
    assert rc == 0 and "not installed" in out and len(_backups(path)) == 2


def test_opencode_install_creates_the_global_config(home, tmp_path, capsys):
    assert _main(capsys, tmp_path, "install", "--client", "opencode")[0] == 0
    path = home / ".config" / "opencode" / "opencode.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"mcp": {"dream": OPENCODE_ENTRY}}
    assert _backups(path) == []
    assert _main(capsys, tmp_path, "uninstall", "--client", "opencode")[0] == 0
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_opencode_honours_xdg_config_home_opencode_config_and_jsonc(home, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert _main(capsys, tmp_path, "install", "--client", "opencode")[0] == 0
    assert json.loads((tmp_path / "xdg" / "opencode" / "opencode.json").read_text())["mcp"]["dream"] == OPENCODE_ENTRY

    jsonc_dir = tmp_path / "xdg2" / "opencode"
    jsonc_dir.mkdir(parents=True)
    (jsonc_dir / "opencode.jsonc").write_text('{"theme": "x"}\n', encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg2"))
    assert _main(capsys, tmp_path, "install", "--client", "opencode")[0] == 0
    assert json.loads((jsonc_dir / "opencode.jsonc").read_text())["mcp"]["dream"] == OPENCODE_ENTRY
    assert not (jsonc_dir / "opencode.json").exists()

    custom = tmp_path / "custom.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(custom))
    assert _main(capsys, tmp_path, "install", "--client", "opencode")[0] == 0
    assert json.loads(custom.read_text())["mcp"]["dream"] == OPENCODE_ENTRY
    assert not (home / ".config").exists()


@pytest.mark.parametrize("content, reason", [
    ("{not json", "not valid JSON"),
    ('{"theme": "x", // comment\n}', "not valid JSON"),
    ('{"theme": "a", "theme": "b"}', "duplicate key"),
    ("[]", "JSON object"),
    ('{"mcp": []}', "mcp must be an object"),
    ('{"mcp": {"dream": {"type": "local", "command": ["someone-else"]}}}', "not written by dream"),
])
def test_opencode_install_refuses_configs_it_cannot_edit_safely(home, tmp_path, capsys, content, reason):
    path = home / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    before = _files(home)
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "opencode")
    assert rc == 1 and out == ""
    assert err.startswith("dream mcp: ") and reason in err
    assert _files(home) == before


def test_opencode_uninstall_leaves_a_foreign_dream_entry(home, tmp_path, capsys):
    path = home / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    foreign = '{"mcp": {"dream": {"type": "remote", "url": "https://example.invalid/mcp"}}}\n'
    path.write_text(foreign, encoding="utf-8")
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "opencode")
    assert rc == 0 and "not written by dream" in out
    assert path.read_text(encoding="utf-8") == foreign and _backups(path) == []


# --- Claude Code --------------------------------------------------------------------------------


class _FakeRun:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.calls = []
        self.result = (returncode, stdout, stderr)

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        returncode, stdout, stderr = self.result
        return mcp_install.subprocess.CompletedProcess(argv, returncode, stdout, stderr)


@pytest.fixture
def claude(monkeypatch):
    fake = _FakeRun(stdout="Added stdio MCP server dream to user config\n")
    monkeypatch.setattr(mcp_install.shutil, "which", lambda name: "/opt/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(mcp_install.subprocess, "run", fake)
    return fake


def test_claude_install_runs_the_documented_command(home, tmp_path, capsys, claude):
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert (rc, err) == (0, "")
    assert [argv for argv, _kwargs in claude.calls] == [
        ["/opt/bin/claude", "mcp", "add", "--scope", "user", "dream", "--", "dream", "mcp", "serve"]]
    kwargs = claude.calls[0][1]
    assert kwargs["capture_output"] is True and kwargs["check"] is False and kwargs["timeout"] > 0
    assert kwargs["stdin"] is mcp_install.subprocess.DEVNULL
    assert "/mcp" in out and "new Claude Code session" in out
    assert _files(home) == {}


def test_claude_install_is_idempotent_when_the_server_exists(home, tmp_path, capsys, claude):
    claude.result = (1, "", "MCP server dream already exists in user config\n")
    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 0 and "already installed" in out


def test_claude_failures_are_reported(home, tmp_path, capsys, claude):
    claude.result = (2, "", "unexpected failure\n")
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and out == "" and "unexpected failure" in err


def test_claude_uninstall_removes_only_the_dream_entry(home, tmp_path, capsys, claude):
    codex = home / ".codex" / "config.toml"
    codex.parent.mkdir()
    codex.write_text(_CODEX_CONFIG, encoding="utf-8")
    before = _files(home)
    claude.result = (0, "Removed MCP server dream from user config\n", "")
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 0 and "Removed" in out
    assert [argv for argv, _kwargs in claude.calls] == [["/opt/bin/claude", "mcp", "remove", "--scope", "user", "dream"]]
    assert _files(home) == before
    claude.result = (1, "", "No user-scoped MCP server found with name: dream\n")
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 0 and "not installed" in out


def test_missing_claude_prints_the_exact_command(home, tmp_path, capsys, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(mcp_install.shutil, "which", lambda name: None)
    monkeypatch.setattr(mcp_install.subprocess, "run", fake)
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and out == ""
    assert "claude mcp add --scope user dream -- dream mcp serve" in err
    rc, out, err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 1 and "claude mcp remove --scope user dream" in err
    assert fake.calls == []


def test_claude_rejects_a_path_option(home, tmp_path, capsys, claude):
    rc, _out, err = _main(capsys, tmp_path, "install", "--client", "claude", "--path", str(tmp_path / "x.json"))
    assert rc == 2 and "--path" in err and claude.calls == []


def test_install_requires_a_known_client(home, tmp_path, capsys):
    for argv in (["install"], ["install", "--client", "vscode"], ["uninstall"]):
        with pytest.raises(SystemExit) as raised:
            _main(capsys, tmp_path, *argv)
        assert raised.value.code == 2


def test_install_works_with_a_broken_dream_configuration(home, tmp_path, capsys):
    from dream import main
    broken = tmp_path / "broken.toml"
    broken.write_text("[recall\n", encoding="utf-8")
    assert main(["--config", str(broken), "mcp", "install", "--client", "codex"]) == 0
    assert tomllib.loads((home / ".codex" / "config.toml").read_text())["mcp_servers"]["dream"] == DREAM_ENTRY
    capsys.readouterr()
