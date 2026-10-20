"""Optional embedding and reranking contracts with deterministic lexical fallback."""

from __future__ import annotations

import http.client
import json
import math
import os
import socket
import ssl
import threading
import urllib.parse
from dataclasses import replace
from typing import Protocol, Sequence, runtime_checkable

from config import has_url_extras, is_loopback_url
from recall_embeddings import EMBED_TEXT_MAX_CODEPOINTS, MAX_DIMENSION, embedding_text, has_vector_blob, store_vector, stored_vector
from recall_types import RecallCandidate, RecallQuery


class AdapterError(RuntimeError):
    pass


class EmbedderUnavailable(AdapterError):
    """The embedding server was unreachable, timed out or answered with an HTTP error."""


@runtime_checkable
class Embedder(Protocol):
    @property
    def fingerprint(self) -> str: ...

    @property
    def remote_data_egress(self) -> bool: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    @property
    def fingerprint(self) -> str: ...

    @property
    def remote_data_egress(self) -> bool: ...

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


def _is_loopback_url(url) -> bool:
    """Loopback by the literal host only (config.is_loopback_url, shared with the
    generation provider); any userinfo, query or fragment makes a URL non-loopback."""
    return isinstance(url, str) and not has_url_extras(url) and is_loopback_url(url)


def _embeddings_endpoint(base_url) -> urllib.parse.SplitResult:
    """Validate a base URL and return its ``/v1/embeddings`` endpoint, split."""
    if not isinstance(base_url, str) or not base_url:
        raise AdapterError("embedder base_url must be a non-empty string")
    try:
        parts = urllib.parse.urlsplit(base_url)
        parts.port  # raises ValueError for a malformed or out-of-range port
    except ValueError:
        raise AdapterError("embedder base_url is not a valid URL") from None
    if parts.scheme not in {"http", "https"}:
        raise AdapterError("embedder base_url must use http or https")
    if "@" in parts.netloc or has_url_extras(base_url):
        raise AdapterError("embedder base_url must not contain credentials, a query or a fragment")
    if not parts.hostname:
        raise AdapterError("embedder base_url has no host")
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path += "/v1"
    return parts._replace(path=path + "/embeddings")


def _short(value, limit: int = 80) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = "".join(character if character.isprintable() else " " for character in text)
    return text[:limit]


# Nightly document batches may take far longer than one query embedding.
BATCH_TIMEOUT_FLOOR_SECONDS = 60.0


class _Aborted(Exception):
    """The caller gave up on this exchange; the worker stops at its next step."""


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


def _read_bounded(response, limit: int, *, raise_on_overflow: bool = True, aborted: threading.Event | None = None) -> bytes:
    """Read at most ``limit`` bytes in small reads; reject a larger body."""
    declared = response.getheader("Content-Length")
    if raise_on_overflow and declared and declared.isdigit() and int(declared) > limit:
        raise AdapterError("embedding response is too large")
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        if aborted is not None and aborted.is_set():
            raise _Aborted()
        chunk = response.read1(min(65536, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
    if total > limit:
        if raise_on_overflow:
            raise AdapterError("embedding response is too large")
        return b"".join(chunks)[:limit]
    return b"".join(chunks)


def _redacted_excerpt(raw: bytes, token) -> str:
    text = raw.decode("utf-8", errors="replace")
    if token:
        text = text.replace(token, "<redacted>")
    return _short(text, 200).strip()


class OpenAICompatibleEmbedder:
    """Embeddings from a local OpenAI-compatible server (llama-server, Ollama).

    One POST per call to ``<base_url>/v1/embeddings`` over ``http.client``:
    no proxies, no redirects, no retries and a bounded response. The timeout
    is a wall-clock bound on the whole exchange (connect, headers and body):
    the request runs in a daemon thread, and when the deadline passes its
    connection is shut down and ``EmbedderUnavailable`` is raised, so a server
    that drips bytes cannot stretch a hook. ``embed`` (queries, preflight)
    uses ``timeout_seconds``; ``embed_documents`` (the nightly write path)
    uses ``batch_timeout_seconds``, by default the same. Any failure raises
    ``AdapterError``; transport failures raise its ``EmbedderUnavailable``
    subclass.
    """

    def __init__(self, base_url, model, dimension, timeout_seconds, api_key_env=None, *, remote_data_egress: bool = False,
                 batch_timeout_seconds=None):
        self._endpoint = _embeddings_endpoint(base_url)
        if not isinstance(model, str) or not model:
            raise AdapterError("embedder model must be a non-empty string")
        if type(dimension) is not int or not 0 < dimension <= MAX_DIMENSION:
            raise AdapterError(f"embedder dimension must be an integer between 1 and {MAX_DIMENSION}")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise AdapterError("embedder timeout_seconds must be a positive number")
        if batch_timeout_seconds is not None and (type(batch_timeout_seconds) not in (int, float)
                                                  or not math.isfinite(batch_timeout_seconds) or batch_timeout_seconds <= 0):
            raise AdapterError("embedder batch timeout must be a positive number")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise AdapterError("embedder api_key_env must name an environment variable")
        loopback = _is_loopback_url(base_url)
        if not loopback and not remote_data_egress:
            raise AdapterError("a non-loopback embedder base_url requires remote_data_egress = true")
        self._model = model
        self._dimension = dimension
        self._timeout = float(timeout_seconds)
        self._batch_timeout = float(batch_timeout_seconds) if batch_timeout_seconds is not None else self._timeout
        self._api_key_env = api_key_env
        self._remote = not loopback
        self.server_model: str | None = None

    @property
    def fingerprint(self) -> str:
        return f"openai-compatible:{self._model}:{self._dimension}"

    @property
    def remote_data_egress(self) -> bool:
        return self._remote

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def _response_limit(self, count: int) -> int:
        # Generous for JSON floats ("-0.0123456789012345," is ~22 bytes).
        return 64 * 1024 + count * self._dimension * 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._post(texts, self._timeout)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._post(texts, self._batch_timeout)

    def _post(self, texts: Sequence[str], timeout: float) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        body = json.dumps({"model": self._model, "input": items, "encoding_format": "float"}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = os.environ.get(self._api_key_env) if self._api_key_env else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        connection = self._connection(timeout)
        aborted = threading.Event()
        outcome: dict = {}

        def exchange() -> None:
            try:
                outcome["raw"] = self._exchange(connection, body, headers, self._response_limit(len(items)), token, aborted)
            except BaseException as exc:  # handed to the caller's thread
                outcome["error"] = exc

        worker = threading.Thread(target=exchange, name="dream-embedder", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            _abort(connection, aborted)
            raise EmbedderUnavailable("embedding server timed out")
        if "error" in outcome:
            raise outcome["error"]
        return self._parse(outcome["raw"], len(items))

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        parts = self._endpoint
        if parts.scheme == "https":
            return http.client.HTTPSConnection(parts.hostname, parts.port, timeout=timeout,
                                               context=ssl.create_default_context())
        return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)

    def _exchange(self, connection, body: bytes, headers: dict, limit: int, token, aborted: threading.Event) -> bytes | None:
        """One request on a fresh connection; transport failures become EmbedderUnavailable.

        The worker checks ``aborted`` after connecting and before every read,
        so it never carries on alone once the caller has given up.
        """
        try:
            connection.connect()
            if aborted.is_set():
                return None
            connection.request("POST", self._endpoint.path, body=body, headers=headers)
            if aborted.is_set():
                return None
            response = connection.getresponse()
            if aborted.is_set():
                return None
            if not 200 <= response.status < 300:
                # Redirects are never followed: any non-2xx answer is unavailability.
                # Read past the excerpt by the key's length so an echoed key is redacted whole.
                raw = _read_bounded(response, 512 + len(token or ""), raise_on_overflow=False, aborted=aborted)
                excerpt = _redacted_excerpt(raw, token)
                raise EmbedderUnavailable(f"embedding server answered HTTP {response.status}" + (f": {excerpt}" if excerpt else ""))
            return _read_bounded(response, limit, aborted=aborted)
        except _Aborted:
            return None
        except AdapterError:
            raise
        except (TimeoutError, socket.timeout):
            raise EmbedderUnavailable("embedding server timed out") from None
        except http.client.HTTPException as exc:
            raise EmbedderUnavailable(f"embedding server sent an invalid HTTP response: {type(exc).__name__}") from None
        except OSError as exc:
            raise EmbedderUnavailable(f"embedding server unreachable: {type(exc).__name__}: {_short(exc.strerror or str(exc))}") from None
        finally:
            connection.close()

    def _parse(self, raw: bytes, count: int) -> list[list[float]]:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise AdapterError("embedding response is not valid JSON") from None
        if not isinstance(payload, dict):
            raise AdapterError("embedding response is not an object")
        if "model" in payload:
            reported = payload["model"]
            if reported != self._model:
                raise AdapterError(f"embedding server reported model {_short(reported)!r}, configured {self._model!r}")
            self.server_model = reported
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != count:
            raise AdapterError("embedding response has the wrong vector count")
        vectors: list[list[float] | None] = [None] * count
        for position, item in enumerate(data):
            if not isinstance(item, dict):
                raise AdapterError("embedding response item is not an object")
            index = item.get("index", position)
            if type(index) is not int or not 0 <= index < count or vectors[index] is not None:
                raise AdapterError("embedding response has an invalid index")
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != self._dimension:
                raise AdapterError(f"embedding response has the wrong dimension (expected {self._dimension})")
            values = []
            for value in embedding:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise AdapterError("embedding response has a non-numeric or non-finite value")
                values.append(float(value))
            vectors[index] = values
        return vectors  # type: ignore[return-value]


def build_embedder(settings) -> OpenAICompatibleEmbedder | None:
    """Build the configured embedder without contacting it; None when it is off.

    Queries keep the configured timeout (1.5 s by default), which bounds a
    session start. Document batches on the write path get at least
    ``BATCH_TIMEOUT_FLOOR_SECONDS``, because a server computes all 32
    embeddings before it answers.
    """
    if not settings.enabled or settings.type == "none":
        return None
    if settings.type != "openai-compatible":
        raise AdapterError(f"unsupported embedder type: {_short(settings.type)!r}")
    timeout = settings.timeout_seconds
    batch_timeout = max(timeout, BATCH_TIMEOUT_FLOOR_SECONDS) if type(timeout) in (int, float) else None
    return OpenAICompatibleEmbedder(
        settings.base_url, settings.model, settings.dimension, timeout, settings.api_key_env,
        remote_data_egress=settings.remote_data_egress, batch_timeout_seconds=batch_timeout,
    )


def adapter_fingerprint(adapter) -> str:
    """The adapter's fingerprint as a string (property or method); AdapterError when missing."""
    value = getattr(adapter, "fingerprint", None)
    value = value() if callable(value) else value
    if not value:
        raise AdapterError("adapter fingerprint is missing")
    return str(value)


def _validate_vector(vector) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or len(values) > MAX_DIMENSION or not all(math.isfinite(value) for value in values):
        raise AdapterError("invalid embedding vector")
    return values


def embed_query(embedder, text: str) -> tuple[float, ...]:
    """Embed one query text with exactly one call."""
    vectors = embedder.embed([text])
    if len(vectors) != 1:
        raise AdapterError("embedder returned the wrong vector count")
    return _validate_vector(vectors[0])


def _cosine(left, right) -> float:
    if len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def embedder_fallback_reason(exc: Exception) -> str:
    return "embedder-unavailable" if isinstance(exc, EmbedderUnavailable) else f"embedder-error:{type(exc).__name__}"


def apply_optional_adapters(conn, query: RecallQuery, candidates: list[RecallCandidate], settings, *, embedder=None, reranker=None, allow_cache_writes: bool = False, query_vector=None, usage=None) -> tuple[list[RecallCandidate], str, str | None]:
    """Enhance lexical candidates, preserving them unchanged on any adapter error.

    The embedder rescores the top 100 candidates that have a stored vector.
    ``query_vector`` reuses a query embedding computed by the caller. Without
    ``allow_cache_writes`` (every read-only path) documents are never embedded:
    a candidate without a stored vector keeps its score, and the mode stays
    ``lexical`` unless at least one candidate was rescored. ``usage`` (passed
    only with ``recall.usage_tie_break``) breaks exact score ties after a
    rescore, as in ``rank_lexical``; without it the order is unchanged.
    """

    def rescored_order(candidate):
        return (-candidate.score, -(usage or {}).get(candidate.document.id, (0,))[0], candidate.document.id)

    current = list(candidates)
    mode = "lexical"
    fallback: str | None = None
    if settings.embedder.enabled:
        if embedder is None:
            fallback = "embedder-unavailable"
        else:
            try:
                fingerprint = adapter_fingerprint(embedder)
                vector_for_query = (_validate_vector(query_vector) if query_vector is not None
                                    else embed_query(embedder, embedding_text(query.query_text)))
                rescored = []
                changed = False
                has_blob = has_vector_blob(conn)
                for candidate in current[:100]:
                    document = candidate.document
                    vector = stored_vector(conn, document.id, document.content_sha256, fingerprint, has_blob=has_blob)
                    if vector is None and allow_cache_writes:
                        vector = embed_query(embedder, candidate.scrubbed_excerpt[:EMBED_TEXT_MAX_CODEPOINTS])
                        store_vector(conn, document.id, document.content_sha256, fingerprint, vector)
                    if vector is None:
                        rescored.append(candidate)
                        continue
                    rescored.append(replace(candidate, score=(candidate.score + max(0.0, _cosine(vector_for_query, vector))) / 2))
                    changed = True
                if allow_cache_writes:
                    conn.commit()
                if changed:
                    current = rescored + current[100:]
                    current.sort(key=rescored_order)
                    mode = "lexical_plus_embedder"
            except Exception as exc:
                fallback = embedder_fallback_reason(exc)
    if settings.reranker.enabled:
        if reranker is None:
            fallback = fallback or "reranker-unavailable"
        else:
            try:
                adapter_fingerprint(reranker)
                values = tuple(float(value) for value in reranker.score(query.query_text, [candidate.scrubbed_excerpt for candidate in current[:100]]))
                if len(values) != min(100, len(current)) or not all(math.isfinite(value) for value in values):
                    raise AdapterError("invalid reranker output")
                current = [replace(candidate, score=(candidate.score + values[index]) / 2) for index, candidate in enumerate(current[:100])] + current[100:]
                current.sort(key=rescored_order)
                mode = "lexical_plus_reranker" if mode == "lexical" else "combined"
            except Exception as exc:
                fallback = fallback or f"reranker-error:{type(exc).__name__}"
    return current, mode, fallback
