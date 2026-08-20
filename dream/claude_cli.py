"""
Headless Claude Code wrapper that uses your OAuth subscription instead of an API key.

We spawn `claude --print --model X --output-format json --json-schema '...' \
                --no-session-persistence --system-prompt '...'` per call, pipe the
prompt on stdin, and parse the resulting JSON. The `structured_output` field
contains the schema-validated response; we never have to strip ```json fences.

Notes on cost / rate limits:
  * Every call carries a ~60-70K-token system-prompt overhead (skills, plugins,
    hooks, memory). Anthropic's prompt cache (1h ephemeral) covers it: the FIRST
    call writes the cache; subsequent calls within ~1h read it back for ~10x
    cheaper effective usage.
  * To benefit from the cache, calls must use IDENTICAL --system-prompt and run
    back-to-back. This module batches them sequentially in a single subprocess
    invocation chain.
  * We run from a dedicated cwd (~/.cache/claude-dream/workdir) so CLAUDE.md
    auto-discovery doesn't pull in random project files.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from model_types import (
    CallResult,
    GenerationResult,
    ProviderError,
    Usage,
    validate_output,
)

# Default workdir for all claude --print invocations. Isolated from any
# CLAUDE.md / per-project memory of cwd from which `dream` was launched.
WORKDIR = Path.home() / ".cache" / "claude-dream" / "workdir"

# Minimal system prompt — replaces the default Claude Code system prompt entirely.
# We still get plugins + skills layered on by the harness; can't eliminate those
# without --bare (which would force API auth). The 60K overhead is what it is;
# cache hits amortize it.
SYSTEM_PROMPT = (
    "You are a memory-distillation worker for a personal Claude Code assistant. "
    "Follow the user's instructions exactly. Output only the JSON object that "
    "satisfies the provided schema — no preamble, no markdown."
)

# Per-call timeout in seconds. Distillation of a long session can take ~30s;
# consolidation with Opus can take longer.
DEFAULT_TIMEOUT_SECS = 600


ClaudeCLIError = ProviderError


def _ensure_workdir() -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    return WORKDIR


def call_claude(
    prompt: str,
    schema: dict,
    model: str | None = "haiku",
    timeout: int = DEFAULT_TIMEOUT_SECS,
    system_prompt: str = SYSTEM_PROMPT,
    extra_args: list[str] | None = None,
    executable: str = "claude",
) -> CallResult:
    """
    Run a single `claude --print` invocation against the user's OAuth subscription.

    `model` accepts an alias ("haiku", "sonnet", "opus") or a full model id.
    `model=None` omits `--model` and lets Claude use its configured default.
    The schema is enforced by Claude Code (`--json-schema`); we then parse the
    `structured_output` from the wrapper JSON.
    """
    workdir = _ensure_workdir()
    schema_json = json.dumps(schema, ensure_ascii=False)

    cmd = [
        executable,
        "--print",
    ]
    if model is not None:
        cmd.extend(["--model", model])
    cmd += [
        "--output-format", "json",
        "--json-schema", schema_json,
        "--no-session-persistence",
        # No-tools worker: consolidate/distill/invent only emit schema-constrained JSON — they never
        # call a tool. Suppress the global MCP config so `claude --print` doesn't spawn every MCP
        # server (Gmail/Calendar/Drive/Pinecone/Sentry) on each call: pure startup overhead + a hang
        # surface for a path that needs zero tools. (--strict-mcp-config = ignore global/project MCP;
        # the empty config = no servers.) The AGENTIC path (call_claude_agent) KEEPS
        # MCP — it genuinely uses tools.
        "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}',
        "--system-prompt", system_prompt,
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    # Don't let the parent session's API key be picked up — we explicitly want OAuth.
    env.pop("ANTHROPIC_API_KEY", None)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=workdir,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"claude CLI timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise ClaudeCLIError(
            f"claude CLI exited {proc.returncode}\nstderr: {proc.stderr[:500]}"
            f"\nstdout: {proc.stdout[:2000]}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise ClaudeCLIError("claude CLI returned empty stdout")

    # `--output-format json` emits one JSON object on stdout (or possibly multiple
    # if multi-turn — we'll take the last well-formed one).
    parsed = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCLIError(f"could not parse claude output: {e}\n--- stdout ---\n{stdout[:2000]}")

    if parsed.get("is_error"):
        raise ClaudeCLIError(f"claude reported error: {parsed.get('result', '')[:500]}")

    structured = parsed.get("structured_output")
    if structured is None:
        # Fallback: try to extract JSON from the text result
        result_text = parsed.get("result", "")
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            raise ClaudeCLIError(
                f"no structured_output and result is not JSON.\nresult: {result_text[:500]}"
            )

    usage = parsed.get("usage", {})
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    validate_output(structured, schema)
    return GenerationResult(
        output=structured,
        raw_result=parsed.get("result", ""),
        provider="claude",
        usage=Usage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_creation_tokens=usage.get("cache_creation_input_tokens"),
            cache_read_tokens=usage.get("cache_read_input_tokens"),
        ),
        total_cost_usd=parsed.get("total_cost_usd", 0.0),
        duration_ms=elapsed_ms,
        model=model,
    )


# --- Streaming status (live "what is the agent doing") -----------------------
#
# When a caller passes on_status to call_claude_agent we run with
# `--output-format stream-json --verbose`, which emits one JSON event per line
# (JSONL): a system/init event, then `assistant` events whose message.content
# carries `tool_use` blocks, `user` events carrying tool_results, and a final
# `result` event with the answer + usage. We map each tool_use to a short
# human label and fire on_status(label); the final answer comes from `result`.

# Plain (built-in) Claude Code tools → user-facing status label.
_TOOL_LABELS: dict[str, str] = {
    "WebSearch": "🔍 szukam w necie…",
    "WebFetch": "🌐 czytam stronę…",
    "Bash": "⚙️ odpalam komendę…",
    "Read": "📄 czytam pliki…",
    "Grep": "🔎 przeszukuję pliki…",
    "Glob": "🔎 przeszukuję pliki…",
    "Edit": "✏️ edytuję plik…",
    "Write": "✏️ zapisuję plik…",
    "NotebookEdit": "✏️ edytuję notebook…",
    "TodoWrite": "📝 planuję…",
    "Task": "🤖 odpalam agenta…",
    "Agent": "🤖 odpalam agenta…",
}

# MCP tool names arrive as mcp__<server>__<tool> (e.g. mcp__claude_ai_Gmail__search_threads).
# Match on the server segment so every tool from that server gets the right label.
_MCP_SERVER_LABELS: list[tuple[str, str]] = [
    ("Gmail", "📧 sprawdzam mail…"),
    ("Calendar", "📅 zerkam w kalendarz…"),
    ("Drive", "📁 grzebię w Drive…"),
    ("pinecone", "🧠 przeszukuję wektory…"),
    ("sentry", "🐞 sprawdzam Sentry…"),
]


def _tool_label(name: str) -> str:
    """Map a tool_use name to a short Polish status label. Defensive: unknown
    tools (incl. future MCP servers) degrade to a generic 'using a tool' label."""
    if name in _TOOL_LABELS:
        return _TOOL_LABELS[name]
    if name.startswith("mcp__"):
        parts = name.split("__")
        server = parts[1] if len(parts) > 1 else ""
        for needle, label in _MCP_SERVER_LABELS:
            if needle.lower() in server.lower():
                return label
        return "🛠️ używam integracji…"
    return "🛠️ używam narzędzia…"


def _consume_stream(lines, on_status: Callable[[str], None] | None) -> dict | None:
    """Consume stream-json (JSONL) events from `lines`, firing on_status(label)
    for each tool_use. Returns the final `result` event dict, or None if the
    stream ended without one. Pure (no subprocess) so it's unit-testable with
    synthetic event lines.

    on_status is best-effort: a raised callback (e.g. a failed UI update)
    is swallowed so a cosmetic status update can never abort the model turn.
    Consecutive identical labels are de-duped at the source to cut edit churn.
    """
    result_event: dict | None = None
    last_label: str | None = None
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "assistant":
            content = (ev.get("message") or {}).get("content") or []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    label = _tool_label(block.get("name", ""))
                    if on_status is not None and label != last_label:
                        last_label = label
                        try:
                            on_status(label)
                        except Exception:
                            pass
        elif etype == "result":
            result_event = ev
    return result_event


def _stream_agent(
    prompt: str,
    model: str,
    append_system_prompt: str | None,
    work: Path,
    timeout: int,
    on_status: Callable[[str], None],
) -> CallResult:
    """Streaming variant of call_claude_agent: same flags but stream-json output,
    so we can report tool use live via on_status. stdin is fed on a background
    thread and stderr to a temp file — both to avoid pipe-buffer deadlocks while
    we read stdout. A watchdog Timer enforces the timeout and is cancelled on
    normal completion so it never kills a reaped/reused PID."""
    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)          # OAuth subscription, never a paid key

    t0 = time.monotonic()
    stderr_f = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_f,
        text=True, bufsize=1, cwd=work, env=env,
    )

    def _feed() -> None:
        try:
            if proc.stdin:
                proc.stdin.write(prompt)
                proc.stdin.close()
        except Exception:
            pass

    timed_out = {"v": False}

    def _kill() -> None:
        timed_out["v"] = True
        proc.kill()

    threading.Thread(target=_feed, daemon=True).start()
    timer = threading.Timer(timeout, _kill)
    timer.start()
    try:
        result_event = _consume_stream(proc.stdout, on_status)
        proc.wait()
    finally:
        timer.cancel()

    if timed_out["v"]:
        raise ClaudeCLIError(f"claude agent timed out after {timeout}s")
    if proc.returncode != 0:
        stderr_f.seek(0)
        err = stderr_f.read()[:500]
        raise ClaudeCLIError(f"claude agent exited {proc.returncode}\nstderr: {err}")
    if result_event is None:
        raise ClaudeCLIError("claude agent stream ended without a result event")
    if result_event.get("is_error"):
        raise ClaudeCLIError(
            f"claude agent reported error: {str(result_event.get('result', ''))[:500]}"
        )

    usage = result_event.get("usage", {}) or {}
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return GenerationResult(
        output={},
        raw_result=result_event.get("result", ""),
        provider="claude",
        usage=Usage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_creation_tokens=usage.get("cache_creation_input_tokens"),
            cache_read_tokens=usage.get("cache_read_input_tokens"),
        ),
        total_cost_usd=result_event.get("total_cost_usd", 0.0),
        duration_ms=elapsed_ms,
        model=model,
    )


def call_claude_agent(
    prompt: str,
    model: str = "sonnet",
    append_system_prompt: str | None = None,
    cwd: Path | None = None,
    timeout: int = 300,
    on_status: Callable[[str], None] | None = None,
) -> CallResult:
    """Agentic `claude --print` turn WITH full tool access (no schema → free-form text).

    Differs from call_claude in three ways, all required for tool use to work:
      * NO --json-schema: output is the model's free-form final answer (parsed["result"]),
        not a schema-constrained object. We return it in CallResult.raw_result.
      * --append-system-prompt (not --system-prompt): keeps Claude Code's default prompt
        intact so the agent still knows how to drive its tools; we only add a persona.
      * --dangerously-skip-permissions: bypass all permission gates so an unattended
        daemon can actually invoke tools (no interactive approver exists). The caller is
        responsible for gating WHO can trigger this (e.g. an owner-only allowlist).

    `cwd` defaults to the dream workdir; pass HOME to give the agent real file/tool reach
    and auto-load ~/CLAUDE.md (the memory index).

    If `on_status` is given, we stream (`--output-format stream-json --verbose`) and
    call it with a short label each time the agent starts a tool, so a caller can show
    live progress. Without it, we keep the simpler blocking `--output-format json` path.
    """
    work = Path(cwd) if cwd is not None else _ensure_workdir()
    work.mkdir(parents=True, exist_ok=True)

    if on_status is not None:
        return _stream_agent(prompt, model, append_system_prompt, work, timeout, on_status)

    cmd = [
        "claude",
        "--print",
        "--model", model,
        "--output-format", "json",
        "--no-session-persistence",
        "--dangerously-skip-permissions",
    ]
    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)          # OAuth subscription, never a paid key

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            cwd=work, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCLIError(f"claude agent timed out after {timeout}s") from e

    if proc.returncode != 0:
        raise ClaudeCLIError(
            f"claude agent exited {proc.returncode}\nstderr: {proc.stderr[:500]}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise ClaudeCLIError("claude agent returned empty stdout")

    parsed = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        raise ClaudeCLIError(f"could not parse claude agent output:\n{stdout[:2000]}")

    if parsed.get("is_error"):
        raise ClaudeCLIError(f"claude agent reported error: {parsed.get('result', '')[:500]}")

    usage = parsed.get("usage", {})
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return GenerationResult(
        output={},
        raw_result=parsed.get("result", ""),
        provider="claude",
        usage=Usage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_creation_tokens=usage.get("cache_creation_input_tokens"),
            cache_read_tokens=usage.get("cache_read_input_tokens"),
        ),
        total_cost_usd=parsed.get("total_cost_usd", 0.0),
        duration_ms=elapsed_ms,
        model=model,
    )


def smoke_test() -> CallResult:
    """Cheap sanity check: returns a hardcoded JSON greeting via Haiku."""
    schema = {
        "type": "object",
        "required": ["greeting"],
        "properties": {"greeting": {"type": "string"}},
    }
    return call_claude("Reply with JSON: {\"greeting\": \"smoke ok\"}", schema, model="haiku")


if __name__ == "__main__":
    r = smoke_test()
    print(f"output={r.output}")
    print(f"in={r.input_tokens} out={r.output_tokens} cache_w={r.cache_creation_tokens} cache_r={r.cache_read_tokens}")
    print(f"cost≈${r.total_cost_usd:.4f}  ({r.duration_ms}ms)")
