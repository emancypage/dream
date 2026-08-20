"""Optional embedding and reranking contracts with deterministic lexical fallback."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Protocol, Sequence, runtime_checkable

from recall_types import RecallCandidate, RecallQuery


class AdapterError(RuntimeError):
    pass


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


def _fingerprint(adapter) -> str:
    value = getattr(adapter, "fingerprint", None)
    value = value() if callable(value) else value
    if not value:
        raise AdapterError("adapter fingerprint is missing")
    return str(value)


def _validate_vector(vector) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or len(values) > 4096 or not all(math.isfinite(value) for value in values):
        raise AdapterError("invalid embedding vector")
    return values


def _cosine(left, right) -> float:
    if len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def apply_optional_adapters(conn, query: RecallQuery, candidates: list[RecallCandidate], settings, *, embedder=None, reranker=None, allow_cache_writes: bool = True) -> tuple[list[RecallCandidate], str, str | None]:
    """Enhance lexical candidates, preserving them unchanged on any adapter error."""
    current = list(candidates)
    mode = "lexical"
    fallback: str | None = None
    if settings.embedder.enabled:
        if embedder is None:
            fallback = "embedder-unavailable"
        else:
            try:
                fingerprint = _fingerprint(embedder)
                query_vector = _validate_vector(embedder.embed([query.query_text])[0])
                vectors = []
                for candidate in current[:100]:
                    row = conn.execute("SELECT vector_json FROM recall_embeddings WHERE document_id=? AND content_sha256=? AND adapter_fingerprint=?", (candidate.document.id, candidate.document.content_sha256, fingerprint)).fetchone()
                    if row:
                        import json
                        vector = _validate_vector(json.loads(row[0]))
                    else:
                        vector = _validate_vector(embedder.embed([candidate.scrubbed_excerpt])[0])
                        if allow_cache_writes:
                            import json
                            conn.execute("INSERT OR REPLACE INTO recall_embeddings(document_id,content_sha256,adapter_fingerprint,vector_json) VALUES(?,?,?,?)", (candidate.document.id, candidate.document.content_sha256, fingerprint, json.dumps(vector)))
                    vectors.append((candidate, vector))
                if allow_cache_writes:
                    conn.commit()
                current = [replace(candidate, score=(candidate.score + max(0.0, _cosine(query_vector, vector))) / 2) for candidate, vector in vectors] + current[100:]
                current.sort(key=lambda candidate: (-candidate.score, candidate.document.id))
                mode = "lexical_plus_embedder"
            except Exception as exc:
                fallback = f"embedder-error:{type(exc).__name__}"
    if settings.reranker.enabled:
        if reranker is None:
            fallback = fallback or "reranker-unavailable"
        else:
            try:
                _fingerprint(reranker)
                values = tuple(float(value) for value in reranker.score(query.query_text, [candidate.scrubbed_excerpt for candidate in current[:100]]))
                if len(values) != min(100, len(current)) or not all(math.isfinite(value) for value in values):
                    raise AdapterError("invalid reranker output")
                current = [replace(candidate, score=(candidate.score + values[index]) / 2) for index, candidate in enumerate(current[:100])] + current[100:]
                current.sort(key=lambda candidate: (-candidate.score, candidate.document.id))
                mode = "lexical_plus_reranker" if mode == "lexical" else "combined"
            except Exception as exc:
                fallback = fallback or f"reranker-error:{type(exc).__name__}"
    return current, mode, fallback
