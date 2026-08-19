"""Strict provider/source configuration with TOML -> env -> CLI precedence."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "dream" / "config.toml"

_TOP_KEYS = {"storage", "stages", "providers", "sources", "review"}
_STORAGE_KEYS = {"db_path", "memory_root"}
_STAGE_KEYS = {"provider", "model", "reasoning_effort", "timeout_seconds"}
_PROVIDER_KEYS = {"type", "auth", "executable", "extra_args"}
_SOURCE_KEYS = {"name", "type", "root", "enabled"}
_REVIEW_KEYS = {"mode", "backup_keep"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DreamConfig:
    data: dict[str, Any]
    path: Path | None

    @property
    def db_path(self) -> Path:
        return Path(self.data["storage"]["db_path"]).expanduser()

    @property
    def memory_root(self) -> Path:
        raw = self.data["storage"]["memory_root"]
        # Keep the packaged default portable across home-directory names.
        raw = raw.replace("-home-szymon", str(Path.home()).replace("/", "-"))
        return Path(raw).expanduser()

    def stage(self, name: str) -> dict[str, Any]:
        try:
            return dict(self.data["stages"][name])
        except KeyError as exc:
            raise ConfigError(f"missing stage configuration: {name}") from exc

    def provider(self, name: str) -> dict[str, Any]:
        try:
            return dict(self.data["providers"][name])
        except KeyError as exc:
            raise ConfigError(f"unknown provider: {name}") from exc

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [dict(source) for source in self.data.get("sources", [])]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _unknown_keys(label: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"unknown {label} options: {', '.join(unknown)}")


def _validate(data: dict[str, Any]) -> None:
    _unknown_keys("top-level", data, _TOP_KEYS)
    _unknown_keys("storage", data.get("storage", {}), _STORAGE_KEYS)
    for name, stage in data.get("stages", {}).items():
        _unknown_keys(f"stage {name}", stage, _STAGE_KEYS)
        if "provider" not in stage or "model" not in stage:
            raise ConfigError(f"stage {name} requires provider and model")
    for name, provider in data.get("providers", {}).items():
        _unknown_keys(f"provider {name}", provider, _PROVIDER_KEYS)
        if provider.get("type") not in {"codex-cli", "claude-cli"}:
            raise ConfigError(f"provider {name} has unsupported type: {provider.get('type')!r}")
    for source in data.get("sources", []):
        _unknown_keys(f"source {source.get('name', '?')}", source, _SOURCE_KEYS)
        if source.get("type") not in {"codex-jsonl", "claude-jsonl"}:
            raise ConfigError(f"unsupported source type: {source.get('type')!r}")
    _unknown_keys("review", data.get("review", {}), _REVIEW_KEYS)
    if data.get("review", {}).get("mode") not in {"suggest-only", "auto-apply"}:
        raise ConfigError("review.mode must be suggest-only or auto-apply")
    providers = data.get("providers", {})
    for name, stage in data.get("stages", {}).items():
        if stage["provider"] not in providers:
            raise ConfigError(f"stage {name} references unknown provider {stage['provider']!r}")


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> DreamConfig:
    default_data = _read_toml(ROOT / "default-config.toml")
    chosen = path or Path(os.environ.get("DREAM_CONFIG", DEFAULT_CONFIG_PATH)).expanduser()
    data = _merge(default_data, _read_toml(chosen)) if chosen.exists() else default_data

    # Legacy compatibility only when no stage-specific setting was supplied.
    legacy_backend = os.environ.get("DREAM_BACKEND")
    if legacy_backend:
        data["stages"]["distill"]["provider"] = legacy_backend
        data["stages"]["consolidate"]["provider"] = legacy_backend
    for stage_name in ("distill", "consolidate"):
        prefix = f"DREAM_{stage_name.upper()}_"
        if os.environ.get(prefix + "PROVIDER"):
            data["stages"][stage_name]["provider"] = os.environ[prefix + "PROVIDER"]
        if os.environ.get(prefix + "MODEL"):
            data["stages"][stage_name]["model"] = os.environ[prefix + "MODEL"]
    if os.environ.get("CLAUDE_DREAM_DB"):
        data["storage"]["db_path"] = os.environ["CLAUDE_DREAM_DB"]
    if os.environ.get("DREAM_MEMORY_ROOT"):
        data["storage"]["memory_root"] = os.environ["DREAM_MEMORY_ROOT"]
    if overrides:
        data = _merge(data, overrides)
    _validate(data)
    return DreamConfig(data=data, path=chosen if chosen.exists() else None)
