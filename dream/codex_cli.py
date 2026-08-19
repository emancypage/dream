"""
Headless Codex CLI provider using `codex exec` and ChatGPT-managed auth.

Selected by the provider-neutral stage router. ChatGPT-managed auth comes from
`codex login`; API-key environment variables are removed for this adapter.

Mechanism (verified against codex-cli 0.135.0):
  * `codex exec` runs non-interactively, prompt piped on stdin (`-` arg).
  * `--output-schema FILE`  enforces the final-message JSON shape (≈ claude's --json-schema).
  * `--output-last-message FILE` writes ONLY the final agent message — clean JSON, no JSONL noise.
  * `--ephemeral`            no session persistence (≈ claude's --no-session-persistence).
  * `--ignore-user-config`   skip ~/.codex/config.toml (MCP plugins, hooks, xhigh default);
                             auth still resolves from CODEX_HOME. We supply model+effort ourselves.
  * `-s read-only`           locked-down sandbox; distillation needs no tools (transcript is in-prompt).

Codex has no --system-prompt; we fold the worker instructions into the prompt body.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from model_types import (
    CallResult,
    GenerationResult,
    ProviderError,
    ensure_workdir,
    validate_output,
)

DEFAULT_TIMEOUT_SECS = 600
SYSTEM_PROMPT = (
    "You are a memory-distillation worker. Follow the instructions exactly and "
    "return only the JSON object required by the supplied schema."
)

# Deprecated aliases retained only for old CLI invocations. New routing passes
# concrete model identifiers and reasoning effort from TOML configuration.
MODEL_MAP = {
    "haiku":  ("gpt-5.4-mini", "low"),
    "sonnet": ("gpt-5.4",      "medium"),
    "opus":   ("gpt-5.5",      "high"),
}


def _resolve_model(model: str) -> tuple[str, str]:
    env = os.environ.get(f"DREAM_CODEX_{model.upper()}")
    if env:
        codex_model, _, effort = env.partition(":")
        return codex_model, (effort or "medium")
    if model in MODEL_MAP:
        return MODEL_MAP[model]
    # Unknown alias → treat the string as a literal codex model id.
    return model, "medium"


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Tolerate a ```json … ``` fence if the model added one despite the schema.
    if raw.startswith("```"):
        body = raw.split("\n", 1)[1] if "\n" in raw else ""
        body = body.rsplit("```", 1)[0]
        try:
            return json.loads(body.strip())
        except json.JSONDecodeError:
            pass
    raise ProviderError(f"could not parse codex output as JSON:\n{raw[:2000]}")


def call_codex(
    prompt: str,
    schema: dict,
    model: str = "haiku",
    timeout: int = DEFAULT_TIMEOUT_SECS,
    system_prompt: str = SYSTEM_PROMPT,
    extra_args: list[str] | None = None,
    reasoning_effort: str | None = None,
    executable: str = "codex",
) -> CallResult:
    """Single `codex exec` invocation against the user's ChatGPT subscription.

    Token usage is unavailable from --output-last-message and is returned as null.
    """
    workdir = ensure_workdir()
    codex_model, effort = _resolve_model(model)
    effort = reasoning_effort or effort
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    with tempfile.TemporaryDirectory() as td:
        schema_path = Path(td) / "schema.json"
        out_path = Path(td) / "last.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

        cmd = [
            executable, "exec",
            "-m", codex_model,
            "-c", f"model_reasoning_effort={effort}",
            "-s", "read-only",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "-C", str(workdir),
            "--output-schema", str(schema_path),
            "-o", str(out_path),
            "-",  # read prompt from stdin
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        # Force ChatGPT-subscription auth; never the paid OpenAI API.
        env.pop("OPENAI_API_KEY", None)
        env.pop("CODEX_API_KEY", None)

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                cwd=workdir,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise ProviderError(f"codex CLI timed out after {timeout}s") from e

        if proc.returncode != 0:
            raise ProviderError(
                f"codex CLI exited {proc.returncode}\nstderr: {proc.stderr[:500]}"
            )

        raw = out_path.read_text(encoding="utf-8").strip() if out_path.exists() else ""

    if not raw:
        raise ProviderError(
            f"codex produced no --output-last-message\nstderr: {proc.stderr[:500]}"
        )

    structured = _parse_json(raw)
    validate_output(structured, schema)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return GenerationResult(
        output=structured,
        raw_result=raw,
        provider="codex",
        usage=None,
        total_cost_usd=None,
        duration_ms=elapsed_ms,
        model=codex_model,
    )


def smoke_test() -> CallResult:
    schema = {
        "type": "object",
        "required": ["greeting"],
        "additionalProperties": False,
        "properties": {"greeting": {"type": "string"}},
    }
    return call_codex('Reply with JSON: {"greeting": "codex smoke ok"}', schema, model="haiku")


if __name__ == "__main__":
    r = smoke_test()
    print(f"output={r.output}  model={r.model}  ({r.duration_ms}ms)")
