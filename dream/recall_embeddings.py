"""Float32 vector storage for recall documents and the pure-Python dense scan.

Vectors are L2-normalized and stored as little-endian float32 blobs in
``recall_embeddings.vector_blob`` with their ``dimension``. ``vector_json``
stays for rows written before the blob columns existed: blob writers store
``'[]'`` there, and readers use it only when the blob is NULL. A database
whose ``recall_embeddings`` table predates the blob columns yields no dense
candidates, so every reader falls back to lexical recall. Nothing here talks
to an embedding server.
"""

from __future__ import annotations

import array
import json
import math
import operator
import sys
from pathlib import Path

from recall_scrub import scrub_text

MAX_DIMENSION = 4096
EMBED_TEXT_MAX_CODEPOINTS = 8000
_BIG_ENDIAN = sys.byteorder == "big"


def embedding_text(text: str) -> str:
    """The text a document is embedded from: scrubbed, then capped."""
    return scrub_text(text, Path.home())[:EMBED_TEXT_MAX_CODEPOINTS]


def normalize_vector(values) -> list[float]:
    """Return the L2-normalized vector; reject empty, oversized, non-finite or zero vectors."""
    vector = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError("embedding values must be numbers")
        vector.append(float(value))
    if not vector or len(vector) > MAX_DIMENSION:
        raise ValueError("embedding has an invalid dimension")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding has a non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm or not math.isfinite(norm):
        raise ValueError("embedding has no direction")
    return [value / norm for value in vector]


def vector_to_blob(values) -> bytes:
    vector = array.array("f", normalize_vector(values))
    if _BIG_ENDIAN:
        vector.byteswap()
    return vector.tobytes()


def blob_to_vector(blob) -> array.array:
    raw = bytes(blob)
    if not raw or len(raw) % 4:
        raise ValueError("embedding blob has an invalid length")
    vector = array.array("f")
    vector.frombytes(raw)
    if _BIG_ENDIAN:
        vector.byteswap()
    return vector


def has_vector_blob(conn) -> bool:
    """Whether ``recall_embeddings`` has the blob columns (read-only safe)."""
    return "vector_blob" in {row[1] for row in conn.execute("PRAGMA table_info(recall_embeddings)")}


def has_vectors(conn, fingerprint: str, allowed_kinds) -> bool:
    """Whether any current document of an allowed kind has a vector under ``fingerprint``.

    Read paths check this before embedding a query, so an unmigrated or
    not yet embedded store never causes a request to the embedding server.
    """
    kinds = tuple(dict.fromkeys(allowed_kinds))
    if not kinds or not has_vector_blob(conn):
        return False
    row = conn.execute(
        "SELECT 1 FROM recall_embeddings e "
        "JOIN recall_documents d ON d.id = e.document_id AND d.content_sha256 = e.content_sha256 "
        "WHERE e.adapter_fingerprint = ? AND (e.vector_blob IS NOT NULL OR e.vector_json != '[]') "
        "AND d.source_kind IN (" + ",".join("?" for _ in kinds) + ") LIMIT 1",
        (fingerprint, *kinds),
    ).fetchone()
    return row is not None


def _json_vector(raw, dimension: int | None = None):
    try:
        values = json.loads(raw) if raw else []
        if not values or (dimension is not None and len(values) != dimension):
            return None
        return array.array("f", normalize_vector(values))
    except (TypeError, ValueError):
        return None


def _blob_vector(blob, dimension):
    if type(dimension) is not int or not 0 < dimension <= MAX_DIMENSION or len(blob) != 4 * dimension:
        return None
    return blob_to_vector(blob)


def store_vector(conn, document_id: str, content_sha256: str, fingerprint: str, values) -> None:
    """Upsert one normalized vector; the caller owns the transaction."""
    blob = vector_to_blob(values)
    conn.execute(
        "INSERT OR REPLACE INTO recall_embeddings(document_id, content_sha256, adapter_fingerprint, vector_json, vector_blob, dimension) "
        "VALUES (?, ?, ?, '[]', ?, ?)",
        (document_id, content_sha256, fingerprint, blob, len(blob) // 4),
    )


def stored_vector(conn, document_id: str, content_sha256: str, fingerprint: str, *, has_blob: bool | None = None):
    """Return the stored vector for one document revision, or None."""
    if has_vector_blob(conn) if has_blob is None else has_blob:
        row = conn.execute(
            "SELECT vector_blob, dimension, vector_json FROM recall_embeddings "
            "WHERE document_id=? AND content_sha256=? AND adapter_fingerprint=?",
            (document_id, content_sha256, fingerprint),
        ).fetchone()
        if row is None:
            return None
        blob, dimension, raw = row
        return _blob_vector(blob, dimension) if blob is not None else _json_vector(raw)
    row = conn.execute(
        "SELECT vector_json FROM recall_embeddings WHERE document_id=? AND content_sha256=? AND adapter_fingerprint=?",
        (document_id, content_sha256, fingerprint),
    ).fetchone()
    return _json_vector(row[0]) if row else None


def dense_candidates(conn, query_vector, fingerprint: str, allowed_kinds, limit: int = 200) -> list[tuple[str, float]]:
    """Scan current-revision vectors of one fingerprint; return positive ``(document_id, cosine)`` descending.

    Only rows whose ``(document_id, content_sha256)`` matches a current
    recall document of an allowed kind are read, so orphaned or stale vectors
    and vectors of another model or dimension are never compared.
    """
    kinds = tuple(dict.fromkeys(allowed_kinds))
    if limit <= 0 or not kinds or not has_vector_blob(conn):
        return []
    query = normalize_vector(query_vector)
    dimension = len(query)
    rows = conn.execute(
        "SELECT e.document_id, e.vector_blob, e.dimension, e.vector_json FROM recall_embeddings e "
        "JOIN recall_documents d ON d.id = e.document_id AND d.content_sha256 = e.content_sha256 "
        "WHERE e.adapter_fingerprint = ? AND d.source_kind IN (" + ",".join("?" for _ in kinds) + ")",
        (fingerprint, *kinds),
    ).fetchall()
    mul = operator.mul
    scored: list[tuple[str, float]] = []
    for document_id, blob, row_dimension, raw in rows:
        if blob is not None:
            if row_dimension != dimension or len(blob) != 4 * dimension:
                continue
            vector = blob_to_vector(blob)
        else:
            vector = _json_vector(raw, dimension)
            if vector is None:
                continue
        # math.sumprod is 3.12+; the runtime supports 3.11.
        cosine = sum(map(mul, query, vector))
        # A vector orthogonal to or pointing away from the query is no evidence.
        if math.isfinite(cosine) and cosine > 0:
            scored.append((document_id, cosine))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]
