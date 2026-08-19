"""Lazy provider construction: unavailable optional adapters stay unimported."""

from __future__ import annotations

import shutil
from pathlib import Path

from model_types import ModelProvider, ProviderError


def create_provider(provider_type: str) -> ModelProvider:
    if provider_type == "codex-cli":
        from providers.codex import CodexCLIProvider

        return CodexCLIProvider()
    if provider_type == "claude-cli":
        from providers.claude import ClaudeCLIProvider

        return ClaudeCLIProvider()
    raise ProviderError(f"unsupported provider type: {provider_type}")


def preflight_provider(config: dict) -> tuple[bool, str]:
    provider_type = config["type"]
    executable = config.get("executable") or {
        "codex-cli": "codex",
        "claude-cli": "claude",
    }.get(provider_type)
    resolved = shutil.which(executable) if executable else None
    if not resolved:
        return False, f"executable not found: {executable}"
    if provider_type == "codex-cli" and config.get("auth") == "chatgpt-subscription":
        auth = Path.home() / ".codex" / "auth.json"
        if not auth.exists():
            return False, f"ChatGPT auth not found: {auth}"
    return True, resolved
