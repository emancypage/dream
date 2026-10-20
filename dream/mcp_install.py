"""Idempotent installation of the Dream MCP server entry in agent clients.

Codex gets a marker-delimited ``[mcp_servers.dream]`` block at the end of
``config.toml``; OpenCode gets ``mcp.dream`` in its global JSON config; Claude
Code is configured through its own ``claude mcp add --scope user`` command.

File edits refuse malformed input, re-parse the result before an atomic
replace, keep a timestamped backup of the previous file and leave every
unrelated entry as it was. Line endings, a UTF-8 byte order mark, the JSON
indentation and the trailing-newline state of the file are kept, so an
install followed by an uninstall restores a Codex file byte for byte.
Installing an identical entry again writes nothing. Uninstall removes only
the entry Dream wrote. Every failure, including an unwritable directory or a
``claude`` that hangs or cannot start, raises ``McpInstallError`` with nothing
written.

The entry runs the ``dream`` launcher found on ``PATH`` by its absolute path,
or ``dream`` by name when none is found. ``DREAM_CONFIG``,
``DREAM_MEMORY_ROOT`` and ``CLAUDE_DREAM_DB``, when set at install time, are
recorded in the client's environment table, so the server reads the same
store. Nothing here starts the server: each report names the client's own
restart or check step.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CLIENTS = ("codex", "opencode", "claude")
SERVER_NAME = "dream"
SERVE_ARGS = ("mcp", "serve")
# Variables that choose Dream's configuration and store; recorded so the client's server sees them too.
RECORDED_ENV = ("DREAM_CONFIG", "DREAM_MEMORY_ROOT", "CLAUDE_DREAM_DB")

CODEX_BEGIN = "# >>> dream mcp server: managed by `dream mcp install`; remove with `dream mcp uninstall` >>>"
CODEX_END = "# <<< dream mcp server <<<"

_CLAUDE_TIMEOUT_SECONDS = 60
_OUTPUT_CODEPOINTS = 400
# Claude Code's own messages for this server; anything else is a failure.
_CLAUDE_EXISTS = re.compile(rf"MCP server ['\"]?{SERVER_NAME}['\"]? already exists", re.IGNORECASE)
_CLAUDE_NOT_FOUND = re.compile(rf"found with name:?\s*['\"]?{SERVER_NAME}['\"]?(?![\w-])", re.IGNORECASE)
_CLAUDE_OURS_INLINE = re.compile(rf"(?:^|[\s/'\"]){SERVER_NAME}['\"]?\s+mcp\s+serve(?![\w-])", re.MULTILINE)
_CLAUDE_COMMAND_LINE = re.compile(rf"^\s*command:\s*(?:\S*/)?{SERVER_NAME}\s*$", re.IGNORECASE | re.MULTILINE)
_CLAUDE_ARGS_LINE = re.compile(r"^\s*args:\s*mcp\s+serve\s*$", re.IGNORECASE | re.MULTILINE)
_SHELL_STOPS = (126, 127)  # the command or its interpreter could not run

_CLIENT_NAMES = {"codex": "Codex", "opencode": "OpenCode", "claude": "Claude Code"}
_NEXT_STEPS = {
    "codex": "Restart Codex so it starts the server, then run /mcp in a new session to check that dream is listed.",
    "opencode": "Restart OpenCode so it starts the server; `opencode mcp list` shows its status.",
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
class ServerLaunch:
    """How the client starts the server: the launcher and the recorded environment."""

    command: str
    on_path: bool
    env: dict[str, str] = field(default_factory=dict)

    @property
    def argv(self) -> list[str]:
        return [self.command, *SERVE_ARGS]


@dataclass(frozen=True)
class McpInstallReport:
    client: str
    changed: bool
    path: str | None
    backup_path: str | None
    detail: str
    notes: tuple[str, ...] = ()


def server_launch() -> ServerLaunch:
    """The ``dream`` launcher on ``PATH`` (absolute) and the Dream variables set now."""
    env = {name: os.environ[name] for name in RECORDED_ENV if os.environ.get(name)}
    found = shutil.which(SERVER_NAME)
    if found:
        return ServerLaunch(os.path.abspath(found), True, env)
    return ServerLaunch(SERVER_NAME, False, env)


def next_step(client: str, launch: ServerLaunch) -> str:
    """The client's own restart or check step, plus the PATH caveat when the launcher was not found."""
    step = _NEXT_STEPS[client]
    if not launch.on_path:
        step += (f" dream was not found on PATH, so the entry runs `dream` by name: {_CLIENT_NAMES[client]} "
                 "must find it on its own PATH.")
    return step


def _env_notes(launch: ServerLaunch, table: str) -> tuple[str, ...]:
    if not launch.env:
        return ()
    return (f"Recorded {', '.join(launch.env)} from the current environment in the entry's {table}.",)


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


@dataclass(frozen=True)
class _Text:
    """A config file's text without its byte order mark, and how to write it back."""

    body: str
    bom: bool
    newline: str

    def encode(self, body: str) -> str:
        return ("\ufeff" if self.bom else "") + body


_NEW_FILE = _Text("", False, "\n")


def _target(path: Path, *, install: bool) -> Path:
    """The file to rewrite: a symlinked config is edited at its target, so the link survives.

    Install refuses a link whose target is missing instead of creating its
    directories; uninstall finds nothing to remove there.
    """
    path = Path(path).expanduser()
    if not path.is_symlink():
        return path
    if not path.exists():
        if install:
            raise McpInstallError(f"{path} is a symbolic link to a missing file; create the target or remove the link first")
        return path
    return path.resolve()


def _read(path: Path, kind: str) -> _Text | None:
    """The file's text, or None when it does not exist. Bytes are decoded as-is, without newline translation."""
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise McpInstallError(f"{path} could not be read: {exc.strerror or type(exc).__name__}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise McpInstallError(f"{path} is not UTF-8 {kind}; fix it first") from exc
    bom = text.startswith("\ufeff")
    body = text[1:] if bom else text
    return _Text(body, bom, "\r\n" if "\r\n" in body else "\n")


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
    """Back up the current file, then atomically replace it, keeping its permission bits.

    ``text`` is written exactly, as UTF-8 bytes without newline translation. A
    read-only file is refused, and any filesystem error becomes
    McpInstallError; the temporary file never outlives a failure.
    """
    if path.exists() and not path.stat().st_mode & stat.S_IWUSR:
        raise McpInstallError(f"{path} is read-only (its owner has no write permission); "
                              "make it writable or edit it by hand")
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise McpInstallError(f"{path} could not be written: the new text is not valid UTF-8; check that "
                              "the dream launcher path and " + ", ".join(RECORDED_ENV) + " are UTF-8") from exc
    temp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = path.stat().st_mode & 0o7777 if path.exists() else None
        backup = _backup(path) if path.exists() else None
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_name, mode)
        os.replace(temp_name, path)
        temp_name = None
        return backup
    except OSError as exc:
        raise McpInstallError(f"{path} could not be written: {exc.strerror or type(exc).__name__}") from exc
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


# --- Codex --------------------------------------------------------------------------------------


def _toml_string(value: str) -> str:
    # JSON escapes quotes, backslashes and C0 controls the way TOML basic strings do. Characters
    # outside the BMP stay literal (TOML has no surrogate-pair escapes), and DEL must be escaped.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _codex_entry(launch: ServerLaunch) -> dict:
    entry = {"command": launch.command, "args": list(SERVE_ARGS)}
    if launch.env:
        entry["env"] = dict(launch.env)
    return entry


def _codex_block(launch: ServerLaunch, newline: str) -> str:
    lines = [CODEX_BEGIN, f"[mcp_servers.{SERVER_NAME}]", f"command = {_toml_string(launch.command)}",
             "args = [" + ", ".join(_toml_string(arg) for arg in SERVE_ARGS) + "]"]
    if launch.env:
        pairs = ", ".join(f"{name} = {_toml_string(value)}" for name, value in launch.env.items())
        lines.append("env = { " + pairs + " }")
    lines.append(CODEX_END)
    return newline.join(lines) + newline


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
        hint = " It contains a U+FEFF byte order mark after the start of the file." if "\ufeff" in text else ""
        if change is None:
            raise McpInstallError(f"{path} is not valid TOML ({exc}); fix it first.{hint}") from exc
        raise McpInstallError(f"{change} would make {path} not valid TOML ({exc}); edit it by hand.{hint}") from exc


def _check_codex_servers(document: dict, path: Path) -> None:
    servers = document.get("mcp_servers")
    if servers is None or isinstance(servers, dict):
        return
    shape = "an array of tables ([[mcp_servers]])" if isinstance(servers, list) else "not a table"
    raise McpInstallError(f"{path}: mcp_servers is {shape}, but Codex expects one [mcp_servers.<name>] table per "
                          "server; fix it first")


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


def install_codex(path: Path | None = None, launch: ServerLaunch | None = None) -> McpInstallReport:
    launch = launch or server_launch()
    path = _target(path or codex_config_path(), install=True)
    current = _read(path, "TOML") or _NEW_FILE
    body, newline = current.body, current.newline
    original = _parse_toml(body, path)
    _check_codex_servers(original, path)
    before, block, after = _split_codex(body, path)
    if block is None and _dream_entry(original) is not None:
        raise McpInstallError(f"{path} already has an [mcp_servers.dream] entry not written by dream; "
                              "remove it or rename it first")
    new_block = _codex_block(launch, newline)
    if block is not None:
        updated = before + new_block + after
    else:
        # One newline ends the last line or adds a blank line; uninstall removes exactly it.
        updated = body + (newline if body else "") + new_block
    parsed = _parse_toml(updated, path, "adding the dream block")
    if _dream_entry(parsed) != _codex_entry(launch):
        raise McpInstallError(f"{path}: keys after the dream block would land in [mcp_servers.dream] with unrelated "
                              "settings; move them above the block or under their own table")
    if _without_dream(parsed) != _without_dream(original):
        raise McpInstallError(f"{path}: adding the dream block would change unrelated settings; edit it by hand")
    notes = _env_notes(launch, "env table")
    if updated == body:
        return McpInstallReport("codex", False, str(path), None, "already installed", notes)
    backup = _replace(path, current.encode(updated))
    return McpInstallReport("codex", True, str(path), backup, "installed" if block is None else "updated", notes)


def uninstall_codex(path: Path | None = None) -> McpInstallReport:
    path = _target(path or codex_config_path(), install=False)
    current = _read(path, "TOML")
    if current is None:
        return McpInstallReport("codex", False, str(path), None, "not installed")
    body, newline = current.body, current.newline
    original = _parse_toml(body, path)
    before, block, after = _split_codex(body, path)
    if block is None:
        detail = ("left an [mcp_servers.dream] entry not written by dream"
                  if _dream_entry(original) is not None else "not installed")
        return McpInstallReport("codex", False, str(path), None, detail)
    # Install added one newline before the block: a blank line, or the end of a last line
    # that had none. Removing it restores the original ending.
    if before.endswith(newline + newline) or (not after and before.endswith(newline)):
        before = before[:-len(newline)]
    updated = before + after
    parsed = _parse_toml(updated, path, "removing the dream block")
    if _dream_entry(parsed) is not None or _without_dream(parsed) != _without_dream(original):
        raise McpInstallError(f"{path}: removing the dream block would change unrelated settings; edit it by hand")
    backup = _replace(path, current.encode(updated))
    return McpInstallReport("codex", True, str(path), backup, "removed")


# --- OpenCode ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _JsonLayout:
    indent: int | str | None
    separators: tuple[str, str] | None
    trailing_newline: bool


_DEFAULT_LAYOUT = _JsonLayout(2, None, True)


def _json_layout(body: str) -> _JsonLayout:
    """The file's indentation unit (spaces or a tab), or compact separators for one-line JSON."""
    if not body.strip():
        return _DEFAULT_LAYOUT
    trailing = body.endswith("\n")
    match = re.search(r"\n([ \t]+)\S", body)
    if match:
        unit = match.group(1)
        return _JsonLayout("\t" if unit.startswith("\t") else len(unit), None, trailing)
    separators = (", ", ": ") if re.search(r'"\s*: ', body) else (",", ":")
    return _JsonLayout(None, separators, trailing)


def _json_text(document: dict, layout: _JsonLayout, newline: str, path: Path) -> str:
    try:
        text = json.dumps(document, ensure_ascii=False, indent=layout.indent, separators=layout.separators,
                          allow_nan=False)
    except ValueError as exc:
        raise McpInstallError(f"{path} contains a number JSON cannot represent (NaN or Infinity); fix it first") from exc
    if layout.trailing_newline:
        text += "\n"
    return text.replace("\n", newline) if newline != "\n" else text


def _no_duplicates(pairs):
    keys = [key for key, _value in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate key {duplicates[0][:40]!r}")
    return dict(pairs)


def _reject_constant(name: str):
    raise ValueError(f"non-standard constant {name}")


def _read_opencode(path: Path) -> tuple[_Text, dict]:
    current = _read(path, "JSON") or _NEW_FILE
    if not current.body.strip():
        return current, {}
    try:
        document = json.loads(current.body, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except ValueError as exc:
        reason = f"has a {exc}" if str(exc).startswith("duplicate key") else f"is not valid JSON ({exc})"
        raise McpInstallError(f"{path} {reason}; comments and trailing commas are not supported, "
                              "so add the entry by hand") from exc
    if not isinstance(document, dict):
        raise McpInstallError(f"{path} must contain a JSON object")
    if "mcp" in document and not isinstance(document["mcp"], dict):
        raise McpInstallError(f"{path}: mcp must be an object")
    return current, document


def _owned_opencode(entry) -> bool:
    """An entry Dream wrote: a local ``<dream launcher> mcp serve`` command."""
    if not isinstance(entry, dict) or entry.get("type") != "local":
        return False
    command = entry.get("command")
    return (isinstance(command, list) and len(command) == 1 + len(SERVE_ARGS) and isinstance(command[0], str)
            and Path(command[0]).name == SERVER_NAME and tuple(command[1:]) == SERVE_ARGS)


def _write_opencode(path: Path, current: _Text, document: dict) -> str | None:
    text = _json_text(document, _json_layout(current.body), current.newline, path)
    if json.loads(text) != document:
        raise McpInstallError(f"{path}: the updated configuration did not round-trip; edit it by hand")
    return _replace(path, current.encode(text))


def install_opencode(path: Path | None = None, launch: ServerLaunch | None = None) -> McpInstallReport:
    launch = launch or server_launch()
    path = _target(path or opencode_config_path(), install=True)
    current, document = _read_opencode(path)
    servers = dict(document.get("mcp") or {})
    existing = servers.get(SERVER_NAME)
    if existing is not None and not _owned_opencode(existing):
        raise McpInstallError(f"{path} already has an mcp.dream entry not written by dream; remove or rename it first")
    # Keys the user added to Dream's entry stay; an explicit enabled: false is respected.
    entry = dict(existing) if existing is not None else {"type": "local", "command": [], "enabled": True}
    entry["type"] = "local"
    entry["command"] = launch.argv
    notes = list(_env_notes(launch, "environment"))
    if entry.get("enabled") is False:
        notes.append("Kept enabled: false on the existing entry; set it to true to start the server.")
    else:
        entry["enabled"] = True
    if launch.env:
        environment = entry.get("environment")
        entry["environment"] = {**(environment if isinstance(environment, dict) else {}), **launch.env}
    if entry == existing:
        return McpInstallReport("opencode", False, str(path), None, "already installed", tuple(notes))
    servers[SERVER_NAME] = entry
    updated = dict(document)
    updated["mcp"] = servers
    backup = _write_opencode(path, current, updated)
    return McpInstallReport("opencode", True, str(path), backup, "installed" if existing is None else "updated",
                            tuple(notes))


def uninstall_opencode(path: Path | None = None) -> McpInstallReport:
    path = _target(path or opencode_config_path(), install=False)
    current, document = _read_opencode(path)
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
    backup = _write_opencode(path, current, updated)
    return McpInstallReport("opencode", True, str(path), backup, "removed")


# --- Claude Code --------------------------------------------------------------------------------


def _claude_add_argv(launch: ServerLaunch) -> list[str]:
    env = [part for name, value in launch.env.items() for part in ("-e", f"{name}={value}")]
    return ["claude", "mcp", "add", "--scope", "user", SERVER_NAME, *env, "--", *launch.argv]


def _claude_display(argv: list[str], launch: ServerLaunch | None = None) -> str:
    """The command as a shell line; recorded variables appear as references, never as values."""
    parts = []
    for part in argv:
        name = part.split("=", 1)[0]
        if launch is not None and "=" in part and name in launch.env and part == f"{name}={launch.env[name]}":
            parts.append(f'{name}="${name}"')
        else:
            parts.append(shlex.quote(part))
    return " ".join(parts)


def _bounded(output: str) -> str:
    text = " ".join(output.split())
    return text if len(text) <= _OUTPUT_CODEPOINTS else text[:_OUTPUT_CODEPOINTS - 1] + "…"


def _run_claude(argv: list[str], launch: ServerLaunch | None = None) -> tuple[int, str]:
    display = _claude_display(argv, launch)
    executable = shutil.which("claude")
    if executable is None:
        raise ClientMissing(display)
    try:
        completed = subprocess.run(
            [executable, *argv[1:]], capture_output=True, text=True, check=False,
            timeout=_CLAUDE_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise McpInstallError(f"`{display}` did not finish within {_CLAUDE_TIMEOUT_SECONDS} s; "
                              "run it yourself to see why") from exc
    except OSError as exc:
        raise McpInstallError(f"`{display}` could not start: {exc.strerror or type(exc).__name__}") from exc
    return completed.returncode, f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()


def _claude_entry_is_ours(output: str) -> bool:
    """Whether ``claude mcp get dream`` describes a ``<dream launcher> mcp serve`` command."""
    if _CLAUDE_OURS_INLINE.search(output):
        return True
    return bool(_CLAUDE_COMMAND_LINE.search(output) and _CLAUDE_ARGS_LINE.search(output))


def _claude_failure(argv: list[str], launch, returncode: int, output: str) -> McpInstallError:
    return McpInstallError(f"`{_claude_display(argv, launch)}` failed with exit code {returncode}: "
                           f"{_bounded(output) or 'no output'}")


def install_claude(launch: ServerLaunch | None = None) -> McpInstallReport:
    """Run ``claude mcp add --scope user dream ...``; Claude Code owns the file and its backup.

    An existing ``dream`` server counts as installed only when ``claude mcp get
    dream`` shows the ``mcp serve`` command; any other failure, including a
    ``claude`` wrapper that cannot run (exit 126 or 127), is an error.
    """
    launch = launch or server_launch()
    argv = _claude_add_argv(launch)
    returncode, output = _run_claude(argv, launch)
    notes = _env_notes(launch, "environment (-e)")
    if returncode == 0:
        return McpInstallReport("claude", True, None, None, "installed", notes)
    if returncode in _SHELL_STOPS or not _CLAUDE_EXISTS.search(output):
        raise _claude_failure(argv, launch, returncode, output)
    get_argv = ["claude", "mcp", "get", SERVER_NAME]
    get_code, get_output = _run_claude(get_argv)
    if get_code == 0 and _claude_entry_is_ours(get_output):
        return McpInstallReport("claude", False, None, None, "already installed", (
            "Kept the existing entry; run `dream mcp uninstall --client claude` and install again to refresh "
            "its command or environment.",))
    raise McpInstallError(f"an MCP server named {SERVER_NAME} already exists; inspect it with `claude mcp get {SERVER_NAME}`")


def uninstall_claude() -> McpInstallReport:
    argv = ["claude", "mcp", "remove", "--scope", "user", SERVER_NAME]
    returncode, output = _run_claude(argv)
    if returncode == 0:
        return McpInstallReport("claude", True, None, None, "removed")
    if returncode not in _SHELL_STOPS and _CLAUDE_NOT_FOUND.search(output):
        return McpInstallReport("claude", False, None, None, "not installed")
    raise _claude_failure(argv, None, returncode, output)


# --- status -------------------------------------------------------------------------------------

# Clients whose entry lives in a file Dream can read; Claude Code keeps its own registry.
FILE_CLIENTS = ("codex", "opencode")


def status(client: str, path: Path | None = None) -> str:
    """Whether a file-based client carries the Dream entry; read-only, for ``dream preflight``.

    Returns ``installed``, ``installed, disabled`` (OpenCode ``"enabled": false``),
    ``not installed`` or ``has an entry not written by dream``. It reads the
    file ``install`` and ``uninstall`` edit, with their parsers, and writes
    nothing. A file it cannot read or parse raises ``McpInstallError``.
    Claude Code is not checked: ``claude mcp get dream`` shows its entry.
    """
    if client == "codex":
        path = _target(path or codex_config_path(), install=False)
        current = _read(path, "TOML")
        if current is None:
            return "not installed"
        document = _parse_toml(current.body, path)
        if _split_codex(current.body, path)[1] is not None:
            return "installed"
        return "has an entry not written by dream" if _dream_entry(document) is not None else "not installed"
    if client == "opencode":
        path = _target(path or opencode_config_path(), install=False)
        _current, document = _read_opencode(path)
        entry = (document.get("mcp") or {}).get(SERVER_NAME)
        if entry is None:
            return "not installed"
        if not _owned_opencode(entry):
            return "has an entry not written by dream"
        return "installed, disabled" if entry.get("enabled") is False else "installed"
    raise McpInstallError(f"no file-based status for client {client!r}; expected one of {', '.join(FILE_CLIENTS)}")


def install(client: str, path: Path | None = None, launch: ServerLaunch | None = None) -> McpInstallReport:
    if client == "codex":
        return install_codex(path, launch)
    if client == "opencode":
        return install_opencode(path, launch)
    if client == "claude":
        return install_claude(launch)
    raise McpInstallError(f"unknown client {client!r}; expected one of {', '.join(CLIENTS)}")


def uninstall(client: str, path: Path | None = None) -> McpInstallReport:
    if client == "codex":
        return uninstall_codex(path)
    if client == "opencode":
        return uninstall_opencode(path)
    if client == "claude":
        return uninstall_claude()
    raise McpInstallError(f"unknown client {client!r}; expected one of {', '.join(CLIENTS)}")
