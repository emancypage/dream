from __future__ import annotations

from codex_cli import call_codex
from model_types import GenerationRequest, GenerationResult


class CodexCLIProvider:
    def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        return call_codex(
            request.prompt,
            request.schema,
            model=request.model,
            timeout=request.timeout_seconds,
            system_prompt=request.system_prompt,
            extra_args=request.options.get("extra_args"),
            reasoning_effort=request.options.get("reasoning_effort"),
            executable=request.options.get("executable") or "codex",
        )
