"""
Headless Codex CLI provider using `codex exec` and ChatGPT-managed auth.

Selected by the provider-neutral stage router. ChatGPT-managed auth comes from
`codex login`; API-key environment variables are removed for this adapter.

Mechanism (verified against codex-cli 0.135.0):
  * `codex exec` runs non-interactively, prompt piped on stdin (`-` arg).
  * `--output-schema FILE`  enforces the final-message JSON shape (≈ claude's --json-schema).
  * `--output-last-message FILE` writes ONLY the final agent message — clean JSON, no JSONL noise.
  * `--json`                 exposes failure events without echoing the input prompt to stderr.
  * `--ephemeral`            no session persistence (≈ claude's --no-session-persistence).
  * `--ignore-user-config`   skip ~/.codex/config.toml (MCP plugins, hooks, xhigh default);
                             auth still resolves from CODEX_HOME. We supply model+effort ourselves.
  * `-s workspace-write`     lets the app-server initialize while restricting model-generated writes
                             to the dedicated Dream workdir from `-C`; memory and database paths
                             are not added as writable directories.

Codex has no --system-prompt; we fold the worker instructions into the prompt body.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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


def _failure_message(stdout: str) -> str | None:
    """Read only error events, never worker output or startup diagnostics."""
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.failed":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else None
        elif event.get("type") == "error":
            message = event.get("message")
        else:
            continue
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _failure_detail(stdout: str, stderr: str) -> str:
    message = _failure_message(stdout)
    if message:
        return f"error: {message[-2000:]}"
    return f"stderr: {stderr.strip()[-2000:]}"


def _temporary_failure(stdout: str) -> bool:
    message = (_failure_message(stdout) or "").casefold()
    statuses = re.findall(
        r"\b(?:unexpected status|http(?: status)?|status(?: code)?)\s*[:=]?\s*(\d{3})\b",
        message,
    )
    if any(status.startswith("4") and status != "429" for status in statuses):
        return False
    if any(term in message for term in (
        "authentication", "unauthorized", "forbidden", "quota", "billing",
        "usage limit", "usage_limit", "spend limit", "spend_limit", "credit balance", "credit_balance",
        "context length", "context_length", "invalid schema", "invalid request",
        "context window", "context_window", "certificate", "refusal", "model refused",
        "not supported", "model_not_found", "does not exist", "sign in", "log in",
        "login", "access token", "refresh token",
    )):
        return False
    return any(status in {"500", "502", "503", "504"} for status in statuses) or any(
        term in message for term in (
            "stream disconnected before completion", "error sending request for url",
            "connection reset", "request timed out", "temporarily overloaded",
            "server_overloaded", "server_is_overloaded", "service unavailable",
            "internal server error", "rate_limit_exceeded", "rate limit reached",
            "rate limit exceeded", "too many requests", "slow_down",
        )
    )


def call_codex(
    prompt: str,
    schema: dict,
    model: str | None = "haiku",
    timeout: int = DEFAULT_TIMEOUT_SECS,
    system_prompt: str = SYSTEM_PROMPT,
    extra_args: list[str] | None = None,
    reasoning_effort: str | None = None,
    executable: str = "codex",
    retry_temporary_errors: bool = False,
) -> CallResult:
    """Invoke `codex exec` against the user's ChatGPT subscription.

    Token usage is unavailable from --output-last-message and is returned as null.
    `model=None` lets Codex use its own default model.
    The summary stage enables one retry for known temporary error events.
    """
    workdir = ensure_workdir()
    if model is None:
        codex_model: str | None = None
        effort: str | None = reasoning_effort
    else:
        codex_model, effort = _resolve_model(model)
        effort = reasoning_effort or effort
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    with tempfile.TemporaryDirectory() as td:
        schema_path = Path(td) / "schema.json"
        out_path = Path(td) / "last.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

        cmd = [executable, "exec"]
        if codex_model:
            cmd.extend(["-m", codex_model])
        if effort:
            cmd.extend(["-c", f"model_reasoning_effort={effort}"])
        cmd += [
            "-s", "workspace-write",
            "--ephemeral",
            "--ignore-user-config",
            "--json",
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
        for attempt in range(2 if retry_temporary_errors else 1):
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

            raw = out_path.read_text(encoding="utf-8").strip() if proc.returncode == 0 and out_path.exists() else ""
            if proc.returncode == 0 and raw:
                break
            detail = _failure_detail(proc.stdout, proc.stderr)
            failure = (
                f"codex CLI exited {proc.returncode}\n{detail}" if proc.returncode != 0
                else f"codex produced no --output-last-message\n{detail}"
            )
            if not retry_temporary_errors or attempt == 1 or proc.returncode < 0 or not _temporary_failure(proc.stdout):
                raise ProviderError(failure)
            print(f"distill: temporary Codex error; retrying once in 5s.\n{detail}", file=sys.stderr)
            out_path.unlink(missing_ok=True)
            time.sleep(5)

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
