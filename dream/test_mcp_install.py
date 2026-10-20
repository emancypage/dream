"""``dream mcp install|uninstall``: idempotent, backed-up, byte-preserving client entries.

Every test patches ``HOME`` and clears ``CODEX_HOME``, ``XDG_CONFIG_HOME``,
``OPENCODE_CONFIG`` and the Dream variables the installer records, so no real
client configuration is read or written. ``PATH`` is a temporary ``bin``
directory plus the system directories, so ``dream`` is not found unless a
test puts a fake launcher there. Claude Code is a fake ``claude`` shell script
in that directory that logs its arguments; no real ``claude`` process starts.
"""

import json
import os
import shutil
import stat
import tomllib
from pathlib import Path

import pytest

import mcp_install
from mcp_install import CODEX_BEGIN, CODEX_END

RECORDED = ("DREAM_CONFIG", "DREAM_MEMORY_ROOT", "CLAUDE_DREAM_DB")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    for name in ("CODEX_HOME", "XDG_CONFIG_HOME", "OPENCODE_CONFIG", *RECORDED):
        monkeypatch.delenv(name, raising=False)
    assert Path.home() == home and shutil.which("dream") is None
    return home


needs_permissions = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                                       reason="root ignores directory permissions")


def _main(capsys, tmp_path, *argv):
    from dream import main
    rc = main(["--config", str(tmp_path / "no-config.toml"), "mcp", *argv])
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _backups(path):
    return sorted(path.parent.glob(path.name + ".bak-*"))


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _fake_dream(tmp_path):
    launcher = tmp_path / "bin" / "dream"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)
    return str(launcher)


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
    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 0 and "Updated the Dream MCP server entry for codex" in out
    assert len(_backups(path)) == 1
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
    ('[[mcp_servers]]\nname = "x"\n', "array of tables ([[mcp_servers]])"),
    ('mcp_servers = "x"\n', "mcp_servers is not a table"),
    ('mcp_servers.dream.command = "d"\n', "not written by dream"),
    ('mcp_servers = { dream = { command = "dream" } }\n', "not written by dream"),
    ('[mcp_servers.dream.env]\nX = "1"\n', "not written by dream"),
    ('model = "x"\n\ufeffother = 1\n', "U+FEFF byte order mark after the start"),
    (b"\xff\xfe m\x00".decode("latin-1"), "not UTF-8"),
])
def test_codex_install_refuses_configs_it_cannot_edit_safely(home, tmp_path, capsys, content, reason):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_bytes(content.encode("latin-1") if reason == "not UTF-8" else content.encode("utf-8"))
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


@pytest.fixture
def claude(home, tmp_path):
    """A fake ``claude`` on PATH: ``claude.set(add=..., get=..., remove=...)`` takes shell snippets."""
    log = tmp_path / "claude-argv.log"
    script = tmp_path / "bin" / "claude"

    class Fake:
        def set(self, add="echo 'Added stdio MCP server dream to user config'", get="exit 1",
                remove="echo 'Removed MCP server dream from user config'"):
            script.write_text(
                "#!/bin/sh\n"
                f"printf '%s\\n' \"$@\" '--end--' >> '{log}'\n"
                f'case "$2" in\n  add) {add} ;;\n  get) {get} ;;\n  remove) {remove} ;;\nesac\n',
                encoding="utf-8")
            script.chmod(0o755)

        def calls(self):
            if not log.exists():
                return []
            calls, current = [], []
            for line in log.read_text(encoding="utf-8").splitlines():
                if line == "--end--":
                    calls.append(current)
                    current = []
                else:
                    current.append(line)
            return calls

    fake = Fake()
    fake.set()
    fake.script = script
    return fake


_ADD = ["mcp", "add", "--scope", "user", "dream", "--", "dream", "mcp", "serve"]
_REMOVE = ["mcp", "remove", "--scope", "user", "dream"]
_EXISTS = "echo 'MCP server dream already exists in user config' >&2; exit 1"
_GET_OURS = ("printf 'dream:\\n  Scope: User config\\n  Status: Connected\\n  Type: stdio\\n  Command: dream\\n"
             "  Args: mcp serve\\n'")
_GET_FOREIGN = ("printf 'dream:\\n  Scope: User config\\n  Type: stdio\\n  Command: npx\\n"
                "  Args: -y dream-dreams-server\\n'")


def test_claude_install_runs_the_documented_command(home, tmp_path, capsys, claude):
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert (rc, err) == (0, "")
    assert claude.calls() == [_ADD]
    assert "/mcp" in out and "new Claude Code session" in out and "dream was not found on PATH" in out
    assert _files(home) == {}


@pytest.mark.parametrize("get, ok", [(_GET_OURS, True), ("printf 'dream: dream mcp serve - Connected\\n'", True),
                                     (_GET_FOREIGN, False), ("echo 'No MCP server found' >&2; exit 1", False)])
def test_claude_install_confirms_an_existing_server_with_mcp_get(home, tmp_path, capsys, claude, get, ok):
    claude.set(add=_EXISTS, get=get)
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert claude.calls() == [_ADD, ["mcp", "get", "dream"]]
    if ok:
        assert rc == 0 and "already installed" in out
    else:
        assert rc == 1 and out == ""
        assert "an MCP server named dream already exists; inspect it with `claude mcp get dream`" in err


@pytest.mark.parametrize("add, code", [
    ("echo 'Error: lock file /x/.claude.json.lock already exists; another process is writing' >&2; exit 1", 1),
    ("echo 'claude: 3: exec: node: not found' >&2; exit 127", 127),
    ("echo 'MCP server dream already exists in user config' >&2; exit 126", 126),
    ("echo 'unexpected failure' >&2; exit 2", 2),
])
def test_claude_install_failures_are_reported_with_the_cli_output(home, tmp_path, capsys, claude, add, code):
    claude.set(add=add)
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and out == ""
    assert f"failed with exit code {code}" in err and len(err) < 700
    assert claude.calls() == [_ADD]


def test_claude_uninstall_removes_only_the_dream_entry(home, tmp_path, capsys, claude):
    codex = home / ".codex" / "config.toml"
    codex.parent.mkdir()
    codex.write_text(_CODEX_CONFIG, encoding="utf-8")
    before = _files(home)
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 0 and "Removed" in out
    assert claude.calls() == [_REMOVE]
    assert _files(home) == before
    claude.set(remove="echo 'No user-scoped MCP server found with name: dream' >&2; exit 1")
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 0 and "not installed" in out


@pytest.mark.parametrize("remove, code", [
    ("echo 'claude: 3: exec: node: not found' >&2; exit 127", 127),
    ("echo 'Error: no credentials found; run claude login' >&2; exit 1", 1),
    ("echo 'No user-scoped MCP server found with name: dream' >&2; exit 127", 127),
    ("echo 'No MCP server found with name: dreamer' >&2; exit 1", 1),
])
def test_claude_uninstall_failures_are_not_reported_as_not_installed(home, tmp_path, capsys, claude, remove, code):
    claude.set(remove=remove)
    rc, out, err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 1 and out == "" and "not installed" not in err
    assert f"failed with exit code {code}" in err


def test_claude_that_hangs_or_cannot_start_is_an_error(home, tmp_path, capsys, claude, monkeypatch):
    monkeypatch.setattr(mcp_install, "_CLAUDE_TIMEOUT_SECONDS", 1)
    claude.set(add="exec sleep 5")
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and out == "" and "did not finish within 1 s" in err and "Traceback" not in err
    claude.script.write_bytes(b"\x7fELFgarbage")
    claude.script.chmod(0o755)
    for action in ("install", "uninstall"):
        rc, out, err = _main(capsys, tmp_path, action, "--client", "claude")
        assert rc == 1 and out == "" and "could not start: Exec format error" in err


def test_missing_claude_prints_the_exact_command(home, tmp_path, capsys):
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and out == ""
    assert "claude mcp add --scope user dream -- dream mcp serve" in err
    rc, out, err = _main(capsys, tmp_path, "uninstall", "--client", "claude")
    assert rc == 1 and "claude mcp remove --scope user dream" in err
    (tmp_path / "bin" / "claude").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "bin" / "claude").chmod(0o644)  # not executable: not a usable claude
    rc, _out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and "not found on PATH" in err


def test_claude_rejects_a_path_option(home, tmp_path, capsys, claude):
    rc, _out, err = _main(capsys, tmp_path, "install", "--client", "claude", "--path", str(tmp_path / "x.json"))
    assert rc == 2 and "--path" in err and claude.calls() == []


# --- launcher and recorded environment ------------------------------------------------------------


def test_the_entry_runs_the_resolved_launcher_by_absolute_path(home, tmp_path, capsys, claude):
    launcher = _fake_dream(tmp_path)
    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 0 and "not found on PATH" not in out
    codex = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    assert codex["mcp_servers"]["dream"] == {"command": launcher, "args": ["mcp", "serve"]}
    assert _main(capsys, tmp_path, "install", "--client", "opencode")[0] == 0
    opencode = json.loads((home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert opencode["mcp"]["dream"]["command"] == [launcher, "mcp", "serve"]
    assert _main(capsys, tmp_path, "install", "--client", "claude")[0] == 0
    assert claude.calls()[-1] == ["mcp", "add", "--scope", "user", "dream", "--", launcher, "mcp", "serve"]
    # Entries written with an absolute launcher are still Dream's own.
    for client in ("codex", "opencode"):
        rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", client)
        assert rc == 0 and "Removed" in out


def test_without_a_launcher_on_path_the_next_step_warns(home, tmp_path, capsys):
    for client, name in (("codex", "Codex"), ("opencode", "OpenCode")):
        rc, out, _err = _main(capsys, tmp_path, "install", "--client", client)
        assert rc == 0
        assert f"dream was not found on PATH, so the entry runs `dream` by name: {name} must find it" in out


def test_dream_variables_are_recorded_by_name_only(home, tmp_path, capsys, claude, monkeypatch):
    values = {"DREAM_CONFIG": str(tmp_path / "secret-config.toml"), "DREAM_MEMORY_ROOT": str(tmp_path / "mem dir"),
              "CLAUDE_DREAM_DB": str(tmp_path / "store" / 'odd"name.db')}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    for client in ("codex", "opencode", "claude"):
        rc, out, err = _main(capsys, tmp_path, "install", "--client", client)
        assert rc == 0, err
        assert "Recorded DREAM_CONFIG, DREAM_MEMORY_ROOT, CLAUDE_DREAM_DB" in out
        assert all(value not in out and value not in err for value in values.values())
    codex = tomllib.loads((home / ".codex" / "config.toml").read_text(encoding="utf-8"))
    assert codex["mcp_servers"]["dream"]["env"] == values
    opencode = json.loads((home / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert opencode["mcp"]["dream"]["environment"] == values
    assert claude.calls()[-1] == ["mcp", "add", "--scope", "user", "dream",
                                  *[part for name, value in values.items() for part in ("-e", f"{name}={value}")],
                                  "--", "dream", "mcp", "serve"]
    # The manual command for a missing claude names the variables without their values.
    claude.script.unlink()
    rc, _out, err = _main(capsys, tmp_path, "install", "--client", "claude")
    assert rc == 1 and '-e DREAM_CONFIG="$DREAM_CONFIG"' in err
    assert all(value not in err for value in values.values())


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


# --- byte preservation ------------------------------------------------------------------------------


_BOM = b"\xef\xbb\xbf"


@pytest.mark.parametrize("original", [
    b"", b'model = "x"', b'model = "x"\n', b'model = "x"\n\n', b'[profiles.fast]\nmodel = "x"',
    b'[mcp_servers.docs]\ncommand = "d"\n', b"[mcp_servers]\n", b'mcp_servers.docs.command = "d"\n',
    b'[mcp_servers]\ndocs.command = "d"\n', b'model = "x"\n# trailing comment', b'model = "x" # c',
    b'note = """\nabc\n"""', b'model = "x"\r\n[features]\r\nmemories = false\r\n', b'model = "x"\r\n[f]\r\nm = 1',
    _BOM + b'model = "x"\n', _BOM + b'model = "x"\r\n[features]\r\nmemories = false\r\n', _BOM,
])
def test_codex_install_and_uninstall_restore_the_file_byte_for_byte(home, tmp_path, original):
    path = tmp_path / "config.toml"
    path.write_bytes(original)
    assert mcp_install.install_codex(path).changed
    installed = path.read_bytes()
    assert tomllib.loads(installed.decode("utf-8").lstrip("﻿"))["mcp_servers"]["dream"] == DREAM_ENTRY
    assert installed.startswith(_BOM) is original.startswith(_BOM)
    if b"\r\n" in original:
        assert installed.count(b"\n") == installed.count(b"\r\n")
    again = mcp_install.install_codex(path)
    assert not again.changed and path.read_bytes() == installed
    assert mcp_install.uninstall_codex(path).changed
    assert path.read_bytes() == original


@pytest.mark.parametrize("indent, newline, bom, trailing", [
    (2, "\n", False, True), (4, "\n", False, True), ("\t", "\n", False, True), (2, "\n", False, False),
    (4, "\r\n", True, True), (None, "\n", False, False), (None, "\n", True, True),
])
def test_opencode_layout_line_endings_and_bom_are_kept(home, tmp_path, indent, newline, bom, trailing):
    document = {"$schema": "https://opencode.ai/config.json", "theme": "tokyonight", "mcp": {"docs": {"type": "remote"}}}
    separators = (",", ":") if indent is None else None
    text = json.dumps(document, indent=indent, separators=separators) + ("\n" if trailing else "")
    original = (_BOM if bom else b"") + text.replace("\n", newline).encode("utf-8")
    path = tmp_path / "opencode.json"
    path.write_bytes(original)
    assert mcp_install.install_opencode(path).changed
    installed = path.read_bytes()
    assert installed.startswith(_BOM) is bom
    body = installed.decode("utf-8").lstrip("﻿")
    assert json.loads(body)["mcp"]["dream"]["command"] == ["dream", "mcp", "serve"]
    assert body.endswith("\n") is trailing
    if newline == "\r\n":
        assert installed.count(b"\n") == installed.count(b"\r\n")
    if indent is not None:
        unit = indent if isinstance(indent, str) else " " * indent
        assert f'\n{unit}"theme"' in body.replace("\r\n", "\n")
    else:
        assert "\n" not in body.rstrip("\n") and '"theme":"tokyonight"' in body
    assert mcp_install.uninstall_opencode(path).changed
    assert path.read_bytes() == original


@pytest.mark.parametrize("content", [b"", b"   \n", _BOM + b"\n"])
def test_an_empty_opencode_file_counts_as_an_empty_object(home, tmp_path, content):
    path = tmp_path / "opencode.json"
    path.write_bytes(content)
    assert mcp_install.install_opencode(path).changed
    body = path.read_bytes().decode("utf-8").lstrip("﻿")
    assert json.loads(body) == {"mcp": {"dream": {"type": "local", "command": ["dream", "mcp", "serve"], "enabled": True}}}
    assert len(_backups(path)) == 1


@pytest.mark.parametrize("content, reason", [
    (b'{"v": NaN}\n', "not valid JSON"),
    (b'{"v": -Infinity}\n', "not valid JSON"),
    (b'{"big": 1e400}\n', "NaN or Infinity"),
])
def test_opencode_numbers_json_cannot_represent_are_refused(home, tmp_path, content, reason):
    path = tmp_path / "opencode.json"
    path.write_bytes(content)
    with pytest.raises(mcp_install.McpInstallError, match=reason):
        mcp_install.install_opencode(path)
    assert path.read_bytes() == content and _backups(path) == []


def test_opencode_reinstall_keeps_user_keys_and_an_explicit_disable(home, tmp_path, capsys):
    path = home / ".config" / "opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    entry = {"type": "local", "command": ["dream", "mcp", "serve"], "enabled": False,
             "environment": {"EXTRA": "1"}, "timeout": 5000}
    path.write_text(json.dumps({"mcp": {"dream": entry}}, indent=2) + "\n", encoding="utf-8")
    original = path.read_bytes()
    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "opencode")
    assert rc == 0 and "already installed" in out and "Kept enabled: false" in out
    assert path.read_bytes() == original and _backups(path) == []

    launcher = _fake_dream(tmp_path)
    rc, out, _err = _main(capsys, tmp_path, "install", "--client", "opencode")
    assert rc == 0 and "Updated the Dream MCP server entry for opencode" in out and "Kept enabled: false" in out
    assert json.loads(path.read_text(encoding="utf-8"))["mcp"]["dream"] == {**entry, "command": [launcher, "mcp", "serve"]}
    assert len(_backups(path)) == 1


# --- filesystem failures --------------------------------------------------------------------------------


@needs_permissions
def test_unwritable_directories_are_reported_without_a_traceback(home, tmp_path, capsys, monkeypatch):
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "x"\n', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    codex_home.chmod(0o555)
    read_only.chmod(0o555)
    try:
        rc, out, err = _main(capsys, tmp_path, "install", "--client", "codex")
        assert rc == 1 and out == "" and "Traceback" not in err
        assert err.startswith("dream mcp: ") and "could not be written: Permission denied" in err
        rc, out, err = _main(capsys, tmp_path, "install", "--client", "opencode",
                             "--path", str(read_only / "opencode.json"))
        assert rc == 1 and "could not be written: Permission denied" in err
    finally:
        codex_home.chmod(0o755)
        read_only.chmod(0o755)
    assert (codex_home / "config.toml").read_text(encoding="utf-8") == 'model = "x"\n'
    assert sorted(path.name for path in codex_home.iterdir()) == ["config.toml"]
    assert list(read_only.iterdir()) == []


def test_a_dangling_symlink_is_refused_without_creating_directories(home, tmp_path, capsys):
    link = home / ".codex" / "config.toml"
    link.parent.mkdir()
    link.symlink_to(tmp_path / "nowhere" / "deep" / "config.toml")
    rc, out, err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 1 and out == "" and "symbolic link to a missing file" in err
    assert not (tmp_path / "nowhere").exists() and link.is_symlink()
    rc, out, _err = _main(capsys, tmp_path, "uninstall", "--client", "codex")
    assert rc == 0 and "not installed" in out


def test_a_symlink_chain_keeps_every_link_and_the_target_mode(home, tmp_path, capsys):
    real = tmp_path / "dotfiles" / "codex.toml"
    real.parent.mkdir()
    real.write_text('model = "x"\n', encoding="utf-8")
    real.chmod(0o640)
    middle = tmp_path / "middle.toml"
    middle.symlink_to(real)
    link = home / ".codex" / "config.toml"
    link.parent.mkdir()
    link.symlink_to(middle)
    assert _main(capsys, tmp_path, "install", "--client", "codex")[0] == 0
    assert link.is_symlink() and middle.is_symlink() and stat.S_IMODE(real.stat().st_mode) == 0o640
    assert [path.name for path in _backups(real)] and not _backups(link)


def test_a_directory_at_the_config_path_is_refused(home, tmp_path, capsys):
    (home / ".codex" / "config.toml").mkdir(parents=True)
    rc, _out, err = _main(capsys, tmp_path, "install", "--client", "codex")
    assert rc == 1 and "could not be read: Is a directory" in err
