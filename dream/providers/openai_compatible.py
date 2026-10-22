"""
OpenAI-compatible chat-completions provider for local model servers.

Targets llama-server, Ollama, vLLM and similar servers that expose
`/v1/chat/completions`, normally on a loopback port. Standard library only;
no streaming and no tool calls.

  * Structured output is requested with `response_format = json_schema`. A
    server that rejects it (HTTP 400/422 with a body naming response_format
    or json_schema) gets exactly one retry with `json_object`, with the
    schema moved into the system prompt. Output is validated host-side
    against the stage schema either way, as for the CLI providers.
  * Every other failure (timeout, refused connection, any other HTTP status,
    unparseable or schema-invalid output) raises ProviderError at once.
  * The stage's `model` and `timeout_seconds` win; the provider's values are
    defaults, and the final timeout fallback is 600 s. backend.generate
    resolves the timeout, this module the model.
  * The timeout is one wall-clock deadline for the whole call: connect,
    headers, the success or error body and the json_object retry. Each
    exchange runs in a daemon thread; when the deadline passes its socket is
    shut down and ProviderError is raised, so a server that drips bytes
    cannot hold a stage past its timeout.
  * Requests go over `http.client`, which reads no proxy variables and
    follows no redirects, so the prompt and the optional key reach only the
    configured host. A non-loopback `base_url` additionally needs
    `remote_data_egress = true`.
  * `api_key_env` names an environment variable read at call time; its value
    is sent as a Bearer header and is never logged, returned or stored. A
    value with control, whitespace or non-ASCII characters is refused before
    any request, naming only the variable.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import ssl
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

from config import has_url_extras, is_loopback_url, is_valid_bearer_token
from model_types import (
    GenerationRequest,
    GenerationResult,
    ProviderError,
    Usage,
    validate_output,
)

PROVIDER = "openai-compatible"
DEFAULT_TIMEOUT_SECONDS = 600
PREFLIGHT_TIMEOUT_SECONDS = 5.0
_RESPONSE_BODY_LIMIT = 16 * 1024 * 1024
_ERROR_BODY_LIMIT = 65536
_ERROR_EXCERPT_CHARS = 300
_READ_CHUNK = 65536


class _Aborted(Exception):
    """The caller gave up on this exchange; the worker stops at its next step."""


class _HTTPStatusError(ProviderError):
    def __init__(self, display: str, status: int, body: str, api_key: str | None = None):
        if api_key:
            body = body.replace(api_key, "[redacted]")  # A server may echo the header.
        excerpt = " ".join(body.split())[:_ERROR_EXCERPT_CHARS]
        super().__init__(f"{PROVIDER} server at {display} returned HTTP {status}: {excerpt}")
        self.status = status
        self.body = body

    def rejects_response_format(self) -> bool:
        body = self.body.lower()
        return self.status in (400, 422) and ("response_format" in body or "json_schema" in body)


class OpenAICompatibleProvider:
    def __init__(self, settings: Mapping[str, Any] | None = None):
        self.settings = dict(settings or {})

    def generate_structured(self, request: GenerationRequest) -> GenerationResult:
        return generate(request, self.settings)


def generate(request: GenerationRequest, settings: Mapping[str, Any]) -> GenerationResult:
    """One chat completion validated against `request.schema`."""
    base_url = _checked_base_url(settings)
    url = _endpoint(base_url, "chat/completions")
    timeout = _timeout(settings, request.timeout_seconds)
    model = request.model or settings.get("model")
    strict = {
        "type": "json_schema",
        "json_schema": {"name": "dream_output", "schema": request.schema, "strict": True},
    }
    t0 = time.monotonic()
    # One deadline for both attempts: the fallback gets only what is left.
    deadline = t0 + timeout
    try:
        data = _request_json("POST", url, _payload(request, model, strict), settings, timeout, deadline)
    except _HTTPStatusError as exc:
        if not exc.rejects_response_format():
            raise
        fallback = _payload(request, model, {"type": "json_object"}, schema_in_prompt=True)
        data = _request_json("POST", url, fallback, settings, timeout, deadline)
    content, finish_reason = _message_content(data)
    output = _parse_json(content, finish_reason)
    validate_output(output, request.schema)
    reported = data.get("model")
    return GenerationResult(
        output=output,
        raw_result=content,
        provider=PROVIDER,
        model=reported if isinstance(reported, str) and reported else model,
        usage=_usage(data),
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def preflight(settings: Mapping[str, Any]) -> tuple[bool, str]:
    """Check reachability and list `/v1/models` without calling the model."""
    try:
        base_url = _checked_base_url(settings)
        timeout = min(PREFLIGHT_TIMEOUT_SECONDS, _timeout(settings, None))
        data = _request_json("GET", _endpoint(base_url, "models"), None, settings, timeout,
                             time.monotonic() + timeout)
    except ProviderError as exc:
        return False, str(exc)
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return False, f"{_display(base_url)} /v1/models returned no model list"
    ids = [entry["id"] for entry in entries if isinstance(entry, dict) and isinstance(entry.get("id"), str)]
    detail = f"{_display(base_url)} reachable; models: {', '.join(ids) or '(none)'}"
    model = settings.get("model")
    if model and model not in ids:
        detail += f"; configured model {model!r} not listed"
    return True, detail


def _checked_base_url(settings: Mapping[str, Any]) -> str:
    base_url = settings.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ProviderError(f"{PROVIDER} provider requires base_url")
    # Config validation enforces both rules too; repeat them for direct callers.
    if has_url_extras(base_url):
        raise ProviderError(f"{PROVIDER} base_url must not contain credentials, a query or a fragment")
    if settings.get("remote_data_egress", False) is not True and not is_loopback_url(base_url):
        raise ProviderError(f"{PROVIDER} base_url is not loopback and remote_data_egress is not true")
    return base_url


def _endpoint(base_url: str, path: str) -> str:
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root += "/v1"
    return f"{root}/{path}"


def _display(base_url: str) -> str:
    # Host and port only: never echo userinfo or paths from the configured URL.
    return urlsplit(base_url).netloc.rpartition("@")[2]


def _timeout(settings: Mapping[str, Any], request_timeout: float | None) -> float:
    return request_timeout or settings.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS


def _payload(
    request: GenerationRequest,
    model: str | None,
    response_format: dict[str, Any],
    *,
    schema_in_prompt: bool = False,
) -> dict[str, Any]:
    system = request.system_prompt
    if schema_in_prompt:
        schema_text = json.dumps(request.schema, sort_keys=True, ensure_ascii=False)
        instruction = f"Return a single JSON object that satisfies this JSON schema:\n{schema_text}"
        system = f"{system}\n\n{instruction}" if system else instruction
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": request.prompt})
    payload: dict[str, Any] = {"messages": messages, "response_format": response_format, "stream": False}
    if model:
        payload["model"] = model
    return payload


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    settings: Mapping[str, Any],
    timeout: float,
    deadline: float,
) -> Any:
    """One request that must finish by ``deadline`` (a ``time.monotonic()`` value).

    ``timeout`` is the call's whole budget, named in the timeout message.
    """
    display = _display(url)
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    key_env = settings.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    if api_key:
        if not is_valid_bearer_token(api_key):
            raise ProviderError(f"api_key_env {key_env} holds a value that is not a valid bearer token "
                                "(control, whitespace or non-ASCII characters)")
        headers["Authorization"] = f"Bearer {api_key}"
    timed_out = ProviderError(f"{PROVIDER} server at {display} timed out after {timeout:g}s")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise timed_out
    parts = urlsplit(url)
    connection = _connection(parts, remaining)
    aborted = threading.Event()
    outcome: dict[str, Any] = {}

    def exchange() -> None:
        try:
            outcome["response"] = _exchange(connection, method, parts.path, body, headers, display, aborted)
        except BaseException as exc:  # handed to the caller's thread
            outcome["error"] = exc

    worker = threading.Thread(target=exchange, name="dream-openai-compatible", daemon=True)
    worker.start()
    worker.join(max(0.0, deadline - time.monotonic()))
    if worker.is_alive():
        _abort(connection, aborted)
        raise timed_out
    if "error" in outcome:
        error = outcome["error"]
        if isinstance(error, (TimeoutError, socket.timeout)):
            raise timed_out from error
        raise error
    status, raw = outcome["response"]
    if not 200 <= status < 300:
        # Redirects are never followed: a 3xx is reported like any other status.
        raise _HTTPStatusError(display, status, raw.decode("utf-8", errors="replace"), api_key)
    if len(raw) > _RESPONSE_BODY_LIMIT:
        raise ProviderError(f"{PROVIDER} response from {display} exceeds {_RESPONSE_BODY_LIMIT} bytes")
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ProviderError(f"{PROVIDER} server at {display} returned a non-JSON response") from exc


def _connection(parts, timeout: float) -> http.client.HTTPConnection:
    if parts.scheme == "https":
        return http.client.HTTPSConnection(parts.hostname, parts.port, timeout=timeout,
                                           context=ssl.create_default_context())
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)


def _exchange(connection, method: str, path: str, body: bytes | None, headers: dict, display: str,
              aborted: threading.Event) -> tuple[int, bytes] | None:
    """One request on a fresh connection: ``(status, body)``, the body read only up to its limit.

    The worker checks ``aborted`` after connecting and before every read, so
    it never carries on alone once the caller has given up. Socket timeouts
    propagate for the caller to report.
    """
    try:
        try:
            connection.connect()
        except (TimeoutError, socket.timeout):
            raise
        except OSError as exc:
            raise ProviderError(f"{PROVIDER} server at {display} is unreachable: "
                                f"{exc.strerror or type(exc).__name__}") from None
        if aborted.is_set():
            return None
        connection.request(method, path or "/", body=body, headers=headers)
        if aborted.is_set():
            return None
        response = connection.getresponse()
        if aborted.is_set():
            return None
        limit = _RESPONSE_BODY_LIMIT + 1 if 200 <= response.status < 300 else _ERROR_BODY_LIMIT
        try:
            raw = _read_bounded(response, limit, aborted)
        except (OSError, http.client.HTTPException):
            if 200 <= response.status < 300:
                raise
            raw = b""  # An unreadable error body still reports the status.
        return response.status, raw
    except _Aborted:
        return None
    except (TimeoutError, socket.timeout):
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise ProviderError(f"{PROVIDER} request to {display} failed: {type(exc).__name__}") from None
    finally:
        connection.close()


def _read_bounded(response, limit: int, aborted: threading.Event) -> bytes:
    """Read at most ``limit`` bytes in small reads, stopping when the caller gives up."""
    chunks: list[bytes] = []
    total = 0
    while total < limit:
        if aborted.is_set():
            raise _Aborted()
        chunk = response.read1(min(_READ_CHUNK, limit - total))
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
    return b"".join(chunks)


def _abort(connection, aborted: threading.Event) -> None:
    """Tell the worker to stop, wake it if it is blocked on the socket, and drop the connection.

    A worker still inside ``connect()`` finishes connecting (bounded by the
    socket timeout), sees the event and closes without sending anything.
    """
    aborted.set()
    sock = getattr(connection, "sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        connection.close()
    except Exception:
        pass


def _message_content(data: Any) -> tuple[str, str | None]:
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{PROVIDER} response has no choices[0].message") from exc
    finish_reason = choice.get("finish_reason")
    if not isinstance(content, str) or not content.strip():
        raise ProviderError(f"{PROVIDER} response has no message content (finish_reason={finish_reason!r})")
    return content, finish_reason


def _parse_json(raw: str, finish_reason: str | None) -> Any:
    raw = raw.strip()
    candidates = [raw]
    # Tolerate a ```json … ``` fence if the model added one despite the format.
    if raw.startswith("```"):
        body = raw.split("\n", 1)[1] if "\n" in raw else ""
        candidates.append(body.rsplit("```", 1)[0].strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # The excerpt is withheld: model output derives from private transcripts.
    raise ProviderError(
        f"could not parse {PROVIDER} output as JSON ({len(raw)} chars, finish_reason={finish_reason!r})"
    )


def _usage(data: dict[str, Any]) -> Usage | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    def count(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return Usage(input_tokens=count("prompt_tokens"), output_tokens=count("completion_tokens"))
