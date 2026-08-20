"""Strict provider/source configuration with TOML -> env -> CLI precedence."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recall_types import AdapterSettings, RecallSettings


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "dream" / "config.toml"

_TOP_KEYS = {"storage", "stages", "providers", "sources", "review", "recall"}
_STORAGE_KEYS = {"db_path", "memory_root"}
_STAGE_KEYS = {"provider", "model", "reasoning_effort", "timeout_seconds"}
_PROVIDER_KEYS = {"type", "auth", "executable", "extra_args"}
_SOURCE_KEYS = {"name", "type", "root", "enabled"}
_REVIEW_KEYS = {"mode", "backup_keep"}
_RECALL_KEYS = {
    "enabled",
    "install_hooks",
    "allow_raw_transcript_prompt",
    "first_prompt_only",
    "session_start_budget_codepoints",
    "prompt_budget_codepoints",
    "session_start_additional_context_limit",
    "prompt_additional_context_limit",
    "diagnostic_path",
    "calibration_path",
    "embedder",
    "reranker",
}
_RECALL_BOOL_KEYS = {"enabled", "install_hooks", "allow_raw_transcript_prompt", "first_prompt_only"}
_RECALL_BUDGET_KEYS = {
    "session_start_budget_codepoints",
    "prompt_budget_codepoints",
    "session_start_additional_context_limit",
    "prompt_additional_context_limit",
}
_RECALL_PATH_KEYS = {"diagnostic_path", "calibration_path"}
_ADAPTER_KEYS = {"enabled", "type", "remote_data_egress"}
_RECALL_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "install_hooks": False,
    "allow_raw_transcript_prompt": False,
    "first_prompt_only": True,
    "session_start_budget_codepoints": 6000,
    "prompt_budget_codepoints": 4000,
    "session_start_additional_context_limit": 1800,
    "prompt_additional_context_limit": 1200,
    "diagnostic_path": "~/.cache/dream/recall-diagnostics.jsonl",
    "calibration_path": "~/.config/dream/recall-calibration.json",
}
_DEFAULT_ADAPTER = AdapterSettings(enabled=False, type="none", remote_data_egress=False)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DreamConfig:
    data: dict[str, Any]
    path: Path | None
    optional_config_errors: tuple[str, ...] = ()

    @property
    def db_path(self) -> Path:
        return Path(self.data["storage"]["db_path"]).expanduser()

    @property
    def memory_root(self) -> Path:
        raw = self.data["storage"]["memory_root"]
        # Keep the packaged default portable across home-directory names.
        raw = raw.replace("__HOME_SLUG__", str(Path.home()).replace("/", "-"))
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

    def recall_settings(self) -> RecallSettings:
        recall = self.data.get("recall")
        recall = recall if isinstance(recall, dict) else {}
        return RecallSettings(
            **{key: recall.get(key, _RECALL_DEFAULTS[key]) for key in _RECALL_DEFAULTS},
            embedder=self._adapter_settings("embedder", recall),
            reranker=self._adapter_settings("reranker", recall),
        )

    @staticmethod
    def _adapter_settings(name: str, recall: dict[str, Any]) -> AdapterSettings:
        raw = recall.get(name)
        if raw is None:
            return _DEFAULT_ADAPTER
        problems: list[str] = []
        _validate_adapter_table(name, raw, problems)
        if problems:
            # An invalid optional adapter falls back to its safe disabled default.
            return _DEFAULT_ADAPTER
        return AdapterSettings(
            enabled=raw.get("enabled", False),
            type=raw.get("type", "none"),
            remote_data_egress=raw.get("remote_data_egress", False),
        )


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


def _validate(data: dict[str, Any]) -> list[str]:
    """Strictly validate core configuration; record optional adapter problems.

    Unknown or mistyped core keys raise ConfigError. Problems in the optional
    [recall.embedder] / [recall.reranker] tables are returned instead, so an
    invalid optional adapter can never block loading the core configuration.
    """
    optional_errors: list[str] = []
    _unknown_keys("top-level", data, _TOP_KEYS)
    _unknown_keys("storage", data.get("storage", {}), _STORAGE_KEYS)
    for name, stage in data.get("stages", {}).items():
        _unknown_keys(f"stage {name}", stage, _STAGE_KEYS)
        if "provider" not in stage:
            raise ConfigError(f"stage {name} requires provider")
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
    _validate_recall(data.get("recall", {}), optional_errors)
    return optional_errors


def _validate_recall(recall: Any, optional_errors: list[str]) -> None:
    if recall is None:
        return
    if not isinstance(recall, dict):
        raise ConfigError("recall must be a table")
    _unknown_keys("recall", recall, _RECALL_KEYS)
    for key in sorted(_RECALL_BOOL_KEYS):
        if key in recall and not isinstance(recall[key], bool):
            raise ConfigError(f"recall.{key} must be a boolean")
    for key in sorted(_RECALL_BUDGET_KEYS):
        value = recall.get(key)
        if key in recall and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ConfigError(f"recall.{key} must be a positive integer")
    for key in sorted(_RECALL_PATH_KEYS):
        value = recall.get(key)
        if key in recall and (not isinstance(value, str) or not value):
            raise ConfigError(f"recall.{key} must be a non-empty string")
    for name in ("embedder", "reranker"):
        if name in recall:
            _validate_adapter_table(name, recall[name], optional_errors)


def _validate_adapter_table(name: str, raw: Any, optional_errors: list[str]) -> None:
    """Record (never raise) problems in one optional adapter table."""
    if not isinstance(raw, dict):
        optional_errors.append(f"recall.{name} must be a table")
        return
    unknown = sorted(set(raw) - _ADAPTER_KEYS)
    if unknown:
        optional_errors.append(f"unknown recall.{name} options: {', '.join(unknown)}")
    for key in ("enabled", "remote_data_egress"):
        if key in raw and not isinstance(raw[key], bool):
            optional_errors.append(f"recall.{name}.{key} must be a boolean")
    if "type" in raw and (not isinstance(raw["type"], str) or not raw["type"]):
        optional_errors.append(f"recall.{name}.type must be a non-empty string")


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
    optional_errors = _validate(data)
    return DreamConfig(
        data=data,
        path=chosen if chosen.exists() else None,
        optional_config_errors=tuple(optional_errors),
    )
