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

_TOP_KEYS = {"storage", "stages", "providers", "sources", "review", "recall", "character"}
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
    "session_start_budget_codepoints": 18000,
    "prompt_budget_codepoints": 4000,
    "session_start_additional_context_limit": 18000,
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


_OPENAI_COMPATIBLE_PROVIDER_KEYS = {"type", "base_url", "model", "api_key_env", "remote_data_egress", "timeout_seconds"}


def is_loopback_url(url: str) -> bool:
    """True only for 127.0.0.0/8, ::1 and the literal name localhost."""
    import ipaddress
    from urllib.parse import urlsplit

    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def has_url_extras(url: str) -> bool:
    """True when a URL carries userinfo, a query or a fragment (all can disguise the host)."""
    from urllib.parse import urlsplit

    # An empty query or fragment ("...:8080?") parses as absent but corrupts the endpoint path.
    if "?" in url or "#" in url:
        return True
    try:
        parts = urlsplit(url)
        return bool(parts.username is not None or parts.password is not None or parts.query or parts.fragment)
    except ValueError:
        return True


def _validate_openai_compatible_provider(name: str, provider: dict[str, Any]) -> None:
    """A local model server; sending prompts off this machine needs explicit consent."""
    from urllib.parse import urlsplit

    _unknown_keys(f"provider {name}", provider, _OPENAI_COMPATIBLE_PROVIDER_KEYS)
    base_url = provider.get("base_url")
    try:
        parts = urlsplit(base_url) if isinstance(base_url, str) else None
        valid_url = bool(parts and parts.scheme in {"http", "https"} and parts.hostname)
        # .port raises ValueError for a malformed or out-of-range port.
        valid_url = valid_url and (parts.port is None or parts.port > 0)
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ConfigError(f"provider {name} requires base_url as an http(s) URL with a host")
    if has_url_extras(base_url):
        raise ConfigError(f"provider {name} base_url must not contain credentials, a query or a fragment")
    for key in ("model", "api_key_env"):
        if key in provider and (not isinstance(provider[key], str) or not provider[key].strip()):
            raise ConfigError(f"provider {name} {key} must be a non-empty string")
    egress = provider.get("remote_data_egress", False)
    if not isinstance(egress, bool):
        raise ConfigError(f"provider {name} remote_data_egress must be a boolean")
    timeout = provider.get("timeout_seconds", 600)
    if type(timeout) is not int or timeout <= 0:
        raise ConfigError(f"provider {name} timeout_seconds must be a positive integer")
    if not egress and not is_loopback_url(base_url):
        raise ConfigError(
            f"provider {name} base_url is not a loopback address; "
            "set remote_data_egress = true to allow sending prompts to it"
        )


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
        if provider.get("type") == "openai-compatible":
            _validate_openai_compatible_provider(name, provider)
            continue
        _unknown_keys(f"provider {name}", provider, _PROVIDER_KEYS)
        if provider.get("type") not in {"codex-cli", "claude-cli"}:
            raise ConfigError(f"provider {name} has unsupported type: {provider.get('type')!r}")
    for source in data.get("sources", []):
        _unknown_keys(f"source {source.get('name', '?')}", source, _SOURCE_KEYS)
        if source.get("type") not in {"codex-jsonl", "claude-jsonl", "opencode-sqlite"}:
            raise ConfigError(f"unsupported source type: {source.get('type')!r}")
    _unknown_keys("review", data.get("review", {}), _REVIEW_KEYS)
    if data.get("review", {}).get("mode") not in {"suggest-only", "auto-apply"}:
        raise ConfigError("review.mode must be suggest-only or auto-apply")
    providers = data.get("providers", {})
    for name, stage in data.get("stages", {}).items():
        if stage["provider"] not in providers:
            raise ConfigError(f"stage {name} references unknown provider {stage['provider']!r}")
    _validate_character(data)
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


def _validate_character(data):
    raw = data.get("character", {})
    if not isinstance(raw, dict):
        raise ConfigError("character must be a table")
    bools = {"enabled", "allow_same_family_judge", "hook_injection"}
    paths = {"constitution_path", "state_root"}
    ints = {"render_max_pairs", "render_budget_codepoints", "max_context_codepoints", "max_response_codepoints", "max_seed_expansions"}
    _unknown_keys("character", raw, bools | paths | ints)
    for key, value in raw.items():
        if key in bools and type(value) is not bool:
            raise ConfigError(f"character.{key} must be a boolean")
        if key in paths and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"character.{key} must be a nonempty path")
        if key in ints and (type(value) is not int or value < (0 if key == "max_seed_expansions" else 1)):
            raise ConfigError(f"character.{key} must be a positive integer")
    constitution = Path(raw["constitution_path"]).expanduser().resolve()
    memory_root = DreamConfig(data, None).memory_root.resolve()
    if constitution.is_relative_to(memory_root):
        raise ConfigError("character constitution_path must stay outside the memory root")
    for name, stage in data.get("stages", {}).items():
        if not name.startswith("character_"):
            continue
        if name not in {"character_extract", "character_generate", "character_judge", "character_eval_response"}:
            raise ConfigError(f"unknown character stage: {name}")
        for key, value in stage.items():
            if key == "timeout_seconds":
                if type(value) is not int or value <= 0:
                    raise ConfigError(f"stage {name} timeout_seconds must be positive integer")
            elif not isinstance(value, str) or not value.strip():
                raise ConfigError(f"stage {name} {key} must be nonempty string")
    if raw.get("enabled"):
        cfg = DreamConfig(data=data, path=None)
        families = {name: provider_family(cfg, name) for name in (
            "character_extract", "character_generate", "character_judge", "character_eval_response"
        )}
        if not raw.get("allow_same_family_judge") and any(
            families[name] == families["character_judge"]
            for name in ("character_extract", "character_generate")
        ):
            raise ConfigError("character judge family must differ from extraction and generation")


def provider_family(config: DreamConfig, stage: str) -> str:
    """Return a known provider/auth family; never infer one from a model name."""
    provider = config.provider(config.stage(stage)["provider"])
    if provider.get("type") == "claude-cli" and "auth" not in provider:
        return "anthropic"
    if provider.get("type") == "codex-cli" and provider.get("auth") == "chatgpt-subscription":
        return "openai"
    raise ConfigError(f"stage {stage} has unknown provider auth/family")
