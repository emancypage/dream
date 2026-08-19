from __future__ import annotations

from claude_cli import call_claude
from model_types import GenerationRequest, GenerationResult


class ClaudeCLIProvider:
    def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        return call_claude(
            request.prompt,
            request.schema,
            model=request.model,
            timeout=request.timeout_seconds,
            system_prompt=request.system_prompt,
            extra_args=request.options.get("extra_args"),
            executable=request.options.get("executable") or "claude",
        )
