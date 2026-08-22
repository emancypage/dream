"""Atomic, selective installation of Dream Codex hook handlers."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

MARKER = "Dream automatic recall v1"


@dataclass(frozen=True)
class HookInstallReport:
    changed: bool
    installed: int
    removed: int
    backup_path: str | None


def validate_hooks_document(document) -> None:
    if not isinstance(document, dict):
        raise ValueError("hooks document must be a JSON object")
    hooks = document.get("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("hooks must be an object")
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            raise ValueError(f"hooks.{event} must be an array")
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks", []), list):
                raise ValueError(f"hooks.{event} contains an invalid group")
            for command in group.get("hooks", []):
                if not isinstance(command, dict):
                    raise ValueError(f"hooks.{event} contains an invalid command")


def _command_name(command_name: str) -> str:
    name = Path(command_name).name
    if not name or name in {".", ".."}:
        raise ValueError("command_name must be a relative executable name")
    return name


def _targets(command_name: str) -> dict[str, tuple[str | None, dict]]:
    name = _command_name(command_name)
    return {
        "SessionStart": (
            "^(startup|resume|clear|compact)$",
            {"type": "command", "command": f"{name} context session-start", "timeout": 5, "statusMessage": MARKER, "additionalContextLimit": 1800},
        ),
        "UserPromptSubmit": (
            None,
            {"type": "command", "command": f"{name} context prompt", "timeout": 5, "statusMessage": MARKER, "additionalContextLimit": 1200},
        ),
    }


def _is_owned(event: str, group: dict, command: dict, target: tuple[str | None, dict]) -> bool:
    matcher, expected = target
    return group.get("matcher") == matcher and command == expected


def _remove_owned(document: dict, command_name: str) -> int:
    targets = _targets(command_name)
    removed = 0
    hooks = document.setdefault("hooks", {})
    for event, target in targets.items():
        groups = hooks.get(event, [])
        kept_groups = []
        for group in groups:
            kept_commands = []
            for command in group.get("hooks", []):
                if _is_owned(event, group, command, target):
                    removed += 1
                else:
                    kept_commands.append(command)
            if kept_commands:
                updated = dict(group)
                updated["hooks"] = kept_commands
                kept_groups.append(updated)
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event, None)
    return removed


def _canonical_groups(command_name: str) -> dict[str, list[dict]]:
    targets = _targets(command_name)
    return {
        event: [{"matcher": matcher, "hooks": [dict(command)]} if matcher is not None else {"hooks": [dict(command)]}]
        for event, (matcher, command) in targets.items()
    }


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid hooks JSON: {exc}") from exc
    validate_hooks_document(document)
    return document


def _backup_and_replace(path: Path, document: dict) -> str | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.name}.bak-{stamp}")
        suffix = 1
        while backup_path.exists():
            backup_path = path.with_name(f"{path.name}.bak-{stamp}-{suffix}")
            suffix += 1
        backup_path.write_bytes(path.read_bytes())
        backup = str(backup_path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return backup


def install_hooks(path: Path, command_name: str = "dream") -> HookInstallReport:
    path = Path(path).expanduser()
    document = _read(path)
    document.setdefault("hooks", {})
    removed = _remove_owned(document, command_name)
    for event, groups in _canonical_groups(command_name).items():
        document["hooks"].setdefault(event, []).extend(groups)
    validate_hooks_document(document)
    backup = _backup_and_replace(path, document)
    return HookInstallReport(True, 2, removed, backup)


def uninstall_hooks(path: Path, command_name: str = "dream") -> HookInstallReport:
    path = Path(path).expanduser()
    document = _read(path)
    removed = _remove_owned(document, command_name)
    if not removed:
        return HookInstallReport(False, 0, 0, None)
    validate_hooks_document(document)
    backup = _backup_and_replace(path, document)
    return HookInstallReport(True, 0, removed, backup)
