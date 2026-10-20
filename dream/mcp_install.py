"""Idempotent installation of the Dream MCP server entry in agent clients.

Codex gets a marker-delimited ``[mcp_servers.dream]`` block at the end of
``config.toml``; OpenCode gets ``mcp.dream`` in its global JSON config; Claude
Code is configured through its own ``claude mcp add --scope user`` command.
File edits refuse malformed input, re-parse the result before an atomic
replace, keep a timestamped backup of the previous file and leave every
unrelated entry as it was. Installing an identical entry again writes
nothing. Uninstall removes only the entry Dream wrote. Nothing here starts
the server: each report names the client's own restart or check step.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

CLIENTS = ("codex", "opencode", "claude")
SERVER_NAME = "dream"
SERVER_COMMAND = ("dream", "mcp", "serve")

CODEX_BEGIN = "# >>> dream mcp server: managed by `dream mcp install`; remove with `dream mcp uninstall` >>>"
CODEX_END = "# <<< dream mcp server <<<"
CODEX_ENTRY = {"command": SERVER_COMMAND[0], "args": list(SERVER_COMMAND[1:])}
OPENCODE_ENTRY = {"type": "local", "command": list(SERVER_COMMAND), "enabled": True}

CLAUDE_ADD = ("claude", "mcp", "add", "--scope", "user", SERVER_NAME, "--", *SERVER_COMMAND)
CLAUDE_REMOVE = ("claude", "mcp", "remove", "--scope", "user", SERVER_NAME)
_CLAUDE_TIMEOUT_SECONDS = 60

NEXT_STEPS = {
    "codex": ("Restart Codex so it starts the server, then run /mcp in a new session to check that dream is "
              "listed. Codex runs `dream mcp serve`, so `dream` must be on the PATH Codex sees."),
    "opencode": ("Restart OpenCode so it starts the server; `opencode mcp list` shows its status. OpenCode runs "
                 "`dream mcp serve`, so `dream` must be on the PATH OpenCode sees."),
    "claude": ("Start a new Claude Code session and run /mcp to check that dream is connected; "
               "`claude mcp get dream` shows the entry."),
}


class McpInstallError(Exception):
    """The client configuration cannot be changed safely; nothing was written."""


class ClientMissing(McpInstallError):
    """The client's executable is not on PATH; ``command`` is the line to run by hand."""

    def __init__(self, command: str):
        super().__init__(f"the claude executable was not found on PATH; run this command yourself:\n  {command}")
        self.command = command


@dataclass(frozen=True)
class McpInstallReport:
    client: str
    changed: bool
    path: str | None
    backup_path: str | None
    detail: str


def codex_config_path() -> Path:
    """``$CODEX_HOME/config.toml``, or ``~/.codex/config.toml`` as Codex itself resolves it."""
    codex_home = os.environ.get("CODEX_HOME")
    return (Path(codex_home).expanduser() if codex_home else Path.home() / ".codex") / "config.toml"


def opencode_config_path() -> Path:
    """OpenCode's global config: ``$OPENCODE_CONFIG``, else ``opencode.json`` or ``opencode.jsonc``.

    The directory is ``$XDG_CONFIG_HOME/opencode`` (default ``~/.config/opencode``).
    An existing ``opencode.json`` wins over ``opencode.jsonc``; without either, a
    new ``opencode.json`` is created.
    """
    custom = os.environ.get("OPENCODE_CONFIG")
    if custom:
        return Path(custom).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    directory = (Path(xdg).expanduser() if xdg else Path.home() / ".config") / "opencode"
    for name in ("opencode.json", "opencode.jsonc"):
        if (directory / name).exists():
            return directory / name
    return directory / "opencode.json"


# --- files --------------------------------------------------------------------------------------


def _target(path: Path) -> Path:
    """The file to rewrite: a symlinked config is edited at its target, so the link survives."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_symlink() else path


def _read_text(path: Path, kind: str) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise McpInstallError(f"{path} could not be read as UTF-8 {kind}: {type(exc).__name__}") from exc


def _backup(path: Path) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak-{stamp}-{suffix}")
        suffix += 1
    shutil.copy2(path, backup)
    return str(backup)


def _replace(path: Path, text: str) -> str | None:
    """Back up the current file, then atomically replace it, keeping its permission bits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o7777 if path.exists() else None
    backup = _backup(path) if path.exists() else None
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return backup


# --- Codex --------------------------------------------------------------------------------------


def _codex_block() -> str:
    return (f"{CODEX_BEGIN}\n[mcp_servers.{SERVER_NAME}]\ncommand = {json.dumps(CODEX_ENTRY['command'])}\n"
            f"args = {json.dumps(CODEX_ENTRY['args'])}\n{CODEX_END}\n")


def _split_codex(text: str, path: Path) -> tuple[str, str | None, str]:
    """``(before, block, after)`` around the one marker block, or ``(text, None, "")``."""
    lines = text.splitlines(keepends=True)
    begins = [index for index, line in enumerate(lines) if line.rstrip() == CODEX_BEGIN]
    ends = [index for index, line in enumerate(lines) if line.rstrip() == CODEX_END]
    if not begins and not ends:
        return text, None, ""
    if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0]:
        raise McpInstallError(f"{path} has incomplete or repeated dream marker lines; fix them by hand")
    start, end = begins[0], ends[0]
    return "".join(lines[:start]), "".join(lines[start:end + 1]), "".join(lines[end + 1:])


def _parse_toml(text: str, path: Path, change: str | None = None) -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        if change is None:
            raise McpInstallError(f"{path} is not valid TOML ({exc}); fix it first") from exc
        raise McpInstallError(f"{change} would make {path} not valid TOML ({exc}); edit it by hand") from exc


def _without_dream(document: dict) -> dict:
    """A copy without ``mcp_servers.dream`` (and without ``mcp_servers`` once it is empty)."""
    copy = dict(document)
    servers = copy.get("mcp_servers")
    if isinstance(servers, dict) and SERVER_NAME in servers:
        servers = {key: value for key, value in servers.items() if key != SERVER_NAME}
        if servers:
            copy["mcp_servers"] = servers
        else:
            copy.pop("mcp_servers")
    elif isinstance(servers, dict) and not servers:
        copy.pop("mcp_servers")
    return copy


def _dream_entry(document: dict):
    servers = document.get("mcp_servers")
    return servers.get(SERVER_NAME) if isinstance(servers, dict) else None


def install_codex(path: Path | None = None) -> McpInstallReport:
    path = _target(path or codex_config_path())
    text = _read_text(path, "TOML") or ""
    original = _parse_toml(text, path)
    before, block, after = _split_codex(text, path)
    if block is None and _dream_entry(original) is not None:
        raise McpInstallError(f"{path} already has an [mcp_servers.dream] entry not written by dream; "
                              "remove it or rename it first")
    if block is not None:
        updated = before + _codex_block() + after
    else:
        separator = "" if not text else ("\n" if text.endswith("\n") else "\n\n")
        updated = text + separator + _codex_block()
    parsed = _parse_toml(updated, path, "adding the dream block")
    if _dream_entry(parsed) != CODEX_ENTRY:
        raise McpInstallError(f"{path}: keys after the dream block would land in [mcp_servers.dream] with unrelated "
                              "settings; move them above the block or under their own table")
    if _without_dream(parsed) != _without_dream(original):
        raise McpInstallError(f"{path}: adding the dream block would change unrelated settings; edit it by hand")
    if updated == text:
        return McpInstallReport("codex", False, str(path), None, "already installed")
    backup = _replace(path, updated)
    return McpInstallReport("codex", True, str(path), backup, "installed")


def uninstall_codex(path: Path | None = None) -> McpInstallReport:
    path = _target(path or codex_config_path())
    text = _read_text(path, "TOML")
    if text is None:
        return McpInstallReport("codex", False, str(path), None, "not installed")
    original = _parse_toml(text, path)
    before, block, after = _split_codex(text, path)
    if block is None:
        detail = ("left an [mcp_servers.dream] entry not written by dream"
                  if _dream_entry(original) is not None else "not installed")
        return McpInstallReport("codex", False, str(path), None, detail)
    if before.endswith("\n\n"):
        before = before[:-1]  # The blank separator line install added.
    updated = before + after
    parsed = _parse_toml(updated, path, "removing the dream block")
    if _dream_entry(parsed) is not None or _without_dream(parsed) != _without_dream(original):
        raise McpInstallError(f"{path}: removing the dream block would change unrelated settings; edit it by hand")
    backup = _replace(path, updated)
    return McpInstallReport("codex", True, str(path), backup, "removed")


# --- OpenCode ---------------------------------------------------------------------------------


def _no_duplicates(pairs):
    keys = [key for key, _value in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate key {duplicates[0][:40]!r}")
    return dict(pairs)


def _read_opencode(path: Path) -> tuple[str | None, dict]:
    text = _read_text(path, "JSON")
    if text is None:
        return None, {}
    try:
        document = json.loads(text, object_pairs_hook=_no_duplicates)
    except ValueError as exc:
        reason = f"has a {exc}" if str(exc).startswith("duplicate key") else f"is not valid JSON ({exc})"
        raise McpInstallError(f"{path} {reason}; comments and trailing commas are not supported, "
                              "so add the entry by hand") from exc
    if not isinstance(document, dict):
        raise McpInstallError(f"{path} must contain a JSON object")
    if "mcp" in document and not isinstance(document["mcp"], dict):
        raise McpInstallError(f"{path}: mcp must be an object")
    return text, document


def _owned_opencode(entry) -> bool:
    return isinstance(entry, dict) and entry.get("type") == "local" and entry.get("command") == list(SERVER_COMMAND)


def _opencode_text(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def install_opencode(path: Path | None = None) -> McpInstallReport:
    path = _target(path or opencode_config_path())
    text, document = _read_opencode(path)
    servers = dict(document.get("mcp") or {})
    if SERVER_NAME in servers and not _owned_opencode(servers[SERVER_NAME]):
        raise McpInstallError(f"{path} already has an mcp.dream entry not written by dream; remove or rename it first")
    if servers.get(SERVER_NAME) == OPENCODE_ENTRY:
        return McpInstallReport("opencode", False, str(path), None, "already installed")
    servers[SERVER_NAME] = dict(OPENCODE_ENTRY)
    updated = dict(document)
    updated["mcp"] = servers
    new_text = _opencode_text(updated)
    if json.loads(new_text) != updated:
        raise McpInstallError(f"{path}: the updated configuration did not round-trip; edit it by hand")
    backup = _replace(path, new_text)
    return McpInstallReport("opencode", True, str(path), backup, "installed")


def uninstall_opencode(path: Path | None = None) -> McpInstallReport:
    path = _target(path or opencode_config_path())
    text, document = _read_opencode(path)
    servers = dict(document.get("mcp") or {})
    if SERVER_NAME not in servers:
        return McpInstallReport("opencode", False, str(path), None, "not installed")
    if not _owned_opencode(servers[SERVER_NAME]):
        return McpInstallReport("opencode", False, str(path), None, "left an mcp.dream entry not written by dream")
    servers.pop(SERVER_NAME)
    updated = dict(document)
    if servers:
        updated["mcp"] = servers
    else:
        updated.pop("mcp", None)
    backup = _replace(path, _opencode_text(updated))
    return McpInstallReport("opencode", True, str(path), backup, "removed")


# --- Claude Code --------------------------------------------------------------------------------


def _run_claude(argv: tuple[str, ...]) -> tuple[int, str]:
    executable = shutil.which("claude")
    if executable is None:
        raise ClientMissing(" ".join(argv))
    completed = subprocess.run(
        [executable, *argv[1:]], capture_output=True, text=True, check=False,
        timeout=_CLAUDE_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
    )
    output = " ".join(f"{completed.stdout or ''} {completed.stderr or ''}".split())
    return completed.returncode, output[:400]


def install_claude() -> McpInstallReport:
    """Run ``claude mcp add --scope user dream -- dream mcp serve``; Claude Code owns the file and its backup."""
    returncode, output = _run_claude(CLAUDE_ADD)
    if returncode == 0:
        return McpInstallReport("claude", True, None, None, "installed")
    if "already exists" in output.casefold():
        return McpInstallReport("claude", False, None, None, "already installed")
    raise McpInstallError(f"`{' '.join(CLAUDE_ADD)}` failed with exit code {returncode}: {output or 'no output'}")


def uninstall_claude() -> McpInstallReport:
    returncode, output = _run_claude(CLAUDE_REMOVE)
    if returncode == 0:
        return McpInstallReport("claude", True, None, None, "removed")
    lowered = output.casefold()
    if "not found" in lowered or ("no " in lowered and "found" in lowered):
        return McpInstallReport("claude", False, None, None, "not installed")
    raise McpInstallError(f"`{' '.join(CLAUDE_REMOVE)}` failed with exit code {returncode}: {output or 'no output'}")


def install(client: str, path: Path | None = None) -> McpInstallReport:
    if client == "codex":
        return install_codex(path)
    if client == "opencode":
        return install_opencode(path)
    if client == "claude":
        return install_claude()
    raise McpInstallError(f"unknown client {client!r}; expected one of {', '.join(CLIENTS)}")


def uninstall(client: str, path: Path | None = None) -> McpInstallReport:
    if client == "codex":
        return uninstall_codex(path)
    if client == "opencode":
        return uninstall_opencode(path)
    if client == "claude":
        return uninstall_claude()
    raise McpInstallError(f"unknown client {client!r}; expected one of {', '.join(CLIENTS)}")
