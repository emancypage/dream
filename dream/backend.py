"""Provider-neutral stage routing for dream model calls."""

from __future__ import annotations

import warnings

from config import DreamConfig, load_config
from model_types import GenerationRequest, GenerationResult
from providers.registry import create_provider
from providers.registry import preflight_provider


SYSTEM_PROMPT = (
    "You are a memory-distillation worker. Follow the instructions exactly and "
    "return only the JSON object required by the supplied schema."
)


def generate(
    stage: str,
    prompt: str,
    schema: dict,
    *,
    model: str | None = None,
    timeout: int | None = None,
    config: DreamConfig | None = None,
) -> GenerationResult:
    cfg = config or load_config()
    stage_cfg = cfg.stage(stage)
    provider_name = stage_cfg["provider"]
    provider_cfg = cfg.provider(provider_name)
    provider = create_provider(provider_cfg["type"])
    options = {
        "reasoning_effort": stage_cfg.get("reasoning_effort"),
        "extra_args": provider_cfg.get("extra_args"),
        "executable": provider_cfg.get("executable"),
    }
    request = GenerationRequest(
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        schema=schema,
        model=model or stage_cfg.get("model"),
        timeout_seconds=timeout or int(stage_cfg.get("timeout_seconds", 600)),
        options={key: value for key, value in options.items() if value is not None},
    )
    return provider.generate_structured(request)


def preflight(config: DreamConfig | None = None) -> list[tuple[str, bool, str]]:
    cfg = config or load_config()
    used = {stage["provider"] for stage in cfg.data["stages"].values()}
    return [
        (name, *preflight_provider(cfg.provider(name)))
        for name in sorted(used)
    ]


def active_backend(stage: str = "distill") -> str:
    """Compatibility helper used by CLI labels."""
    return load_config().stage(stage)["provider"]


def call_claude(prompt, schema, model="haiku", **kwargs):
    """Deprecated compatibility shim for callers not yet migrated to generate()."""
    warnings.warn("call_claude is deprecated; use backend.generate", DeprecationWarning, stacklevel=2)
    stage = "consolidate" if model == "opus" else "distill"
    return generate(stage, prompt, schema, model=model, timeout=kwargs.get("timeout"))
