"""Float32 vector storage, the dense scan, the vector-column migration and nightly embedding."""

import array
import math
import random
import sqlite3
import struct
import time

import pytest

from recall_documents import embed_recall_documents, synchronize_recall_documents
from recall_embeddings import (
    EMBED_TEXT_MAX_CODEPOINTS,
    blob_to_vector,
    dense_candidates,
    embedding_text,
    has_vector_blob,
    store_vector,
    stored_vector,
    vector_to_blob,
)

FINGERPRINT = "openai-compatible:fake:3"
KINDS = ("approved_memory", "distilled_summary")


class RecordingEmbedder:
    fingerprint = FINGERPRINT
    remote_data_egress = False

    def __init__(self, fail_after=None):
        self.batches = []
        self.fail_after = fail_after

    def embed(self, texts):
        if self.fail_after is not None and len(self.batches) >= self.fail_after:
            raise RuntimeError("offline")
        self.batches.append(list(texts))
        return [[1.0, float(len(text) % 7), 0.5] for text in texts]


def _db(tmp_path):
    from dream import open_db
    return open_db(tmp_path / "dream.db")


def _doc(conn, doc_id, sha=None, kind="approved_memory", text=None):
    trust = {"approved_memory": "user_approved", "distilled_summary": "model_distilled", "raw_transcript": "untrusted_transcript"}[kind]
    conn.execute(
        "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, source_updated_at, indexed_at, source_version, text) "
        "VALUES (?, ?, ?, ?, NULL, ?, '', '', 'v1', ?)",
        (doc_id, sha or f"sha-{doc_id}", kind, trust, f"{doc_id}.md", text or f"text of {doc_id}"),
    )


def test_blob_round_trip_is_normalized_little_endian_float32():
    blob = vector_to_blob([3.0, 4.0])
    assert len(blob) == 8
    assert struct.unpack("<2f", blob) == pytest.approx((0.6, 0.8))
    vector = blob_to_vector(blob)
    assert isinstance(vector, array.array) and vector.typecode == "f"
    assert list(vector) == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize("values", [[], [0.0, 0.0], [1.0, float("nan")], [float("inf")], [1.0] * 4097, ["x"], [True, 1.0]])
def test_invalid_vectors_are_rejected(values):
    with pytest.raises(ValueError):
        vector_to_blob(values)


def test_invalid_blobs_are_rejected():
    with pytest.raises(ValueError):
        blob_to_vector(b"")
    with pytest.raises(ValueError):
        blob_to_vector(b"\x00\x00\x80")


def test_embedding_text_is_scrubbed_and_capped():
    text = "OPENAI_API_KEY=sk-secret-value\n" + "x" * 20000
    value = embedding_text(text)
    assert "sk-secret-value" not in value and "<redacted>" in value
    assert len(value) == EMBED_TEXT_MAX_CODEPOINTS


def test_dense_candidates_rank_current_documents_by_cosine(tmp_path):
    conn = _db(tmp_path)
    for doc_id in ("a", "b", "c", "d"):
        _doc(conn, doc_id)
    _doc(conn, "s", kind="distilled_summary")
    store_vector(conn, "a", "sha-a", FINGERPRINT, [1.0, 0.0, 0.0])
    store_vector(conn, "b", "sha-b", FINGERPRINT, [0.6, 0.8, 0.0])
    store_vector(conn, "c", "sha-c", FINGERPRINT, [0.6, 0.8, 0.0])
    store_vector(conn, "d", "sha-d", FINGERPRINT, [0.0, 0.0, 1.0])
    store_vector(conn, "s", "sha-s", FINGERPRINT, [0.9, 0.1, 0.0])
    conn.commit()
    results = dense_candidates(conn, [2.0, 0.0, 0.0], FINGERPRINT, KINDS)
    # d is orthogonal to the query: no evidence, no candidate.
    assert [doc_id for doc_id, _ in results] == ["a", "s", "b", "c"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[2][1] == pytest.approx(0.6)
    assert dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, KINDS, limit=2) == results[:2]
    assert [doc_id for doc_id, _ in dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, ("distilled_summary",))] == ["s"]
    assert dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, ()) == []


def test_vectors_under_another_fingerprint_or_dimension_are_never_compared(tmp_path, monkeypatch):
    import recall_embeddings
    conn = _db(tmp_path)
    _doc(conn, "a")
    _doc(conn, "b")
    store_vector(conn, "a", "sha-a", "openai-compatible:old-model:3", [1.0, 0.0, 0.0])
    store_vector(conn, "b", "sha-b", FINGERPRINT, [1.0, 0.0, 0.0, 0.0])
    conn.commit()
    compared = []
    original = recall_embeddings.blob_to_vector
    monkeypatch.setattr(recall_embeddings, "blob_to_vector", lambda blob: compared.append(blob) or original(blob))
    assert dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, KINDS) == []
    assert compared == []


def test_orphaned_and_stale_revision_vectors_never_become_candidates(tmp_path):
    conn = _db(tmp_path)
    _doc(conn, "a", sha="sha-new")
    store_vector(conn, "a", "sha-old", FINGERPRINT, [1.0, 0.0, 0.0])
    store_vector(conn, "gone", "sha-gone", FINGERPRINT, [1.0, 0.0, 0.0])
    conn.commit()
    assert dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, KINDS) == []


def test_legacy_json_vectors_are_read_when_the_blob_is_missing(tmp_path):
    conn = _db(tmp_path)
    _doc(conn, "a")
    _doc(conn, "b")
    conn.execute("INSERT INTO recall_embeddings(document_id, content_sha256, adapter_fingerprint, vector_json) VALUES ('a', 'sha-a', ?, '[0.0, 2.0, 0.0]')", (FINGERPRINT,))
    conn.execute("INSERT INTO recall_embeddings(document_id, content_sha256, adapter_fingerprint, vector_json) VALUES ('b', 'sha-b', ?, '[]')", (FINGERPRINT,))
    conn.commit()
    assert dense_candidates(conn, [0.0, 1.0, 0.0], FINGERPRINT, KINDS) == [("a", pytest.approx(1.0))]
    assert list(stored_vector(conn, "a", "sha-a", FINGERPRINT)) == pytest.approx([0.0, 1.0, 0.0])
    assert stored_vector(conn, "b", "sha-b", FINGERPRINT) is None


def test_blob_with_wrong_length_is_skipped(tmp_path):
    conn = _db(tmp_path)
    _doc(conn, "a")
    conn.execute("INSERT INTO recall_embeddings(document_id, content_sha256, adapter_fingerprint, vector_json, vector_blob, dimension) VALUES ('a', 'sha-a', ?, '[]', ?, 3)",
                 (FINGERPRINT, vector_to_blob([1.0, 0.0])))
    conn.commit()
    assert dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, KINDS) == []
    assert stored_vector(conn, "a", "sha-a", FINGERPRINT) is None


def _unmigrated(tmp_path):
    from dream import ROOT
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    return conn


def test_database_without_vector_columns_fails_open(tmp_path):
    conn = _unmigrated(tmp_path)
    assert not has_vector_blob(conn)
    _doc(conn, "a")
    conn.execute("INSERT INTO recall_embeddings(document_id, content_sha256, adapter_fingerprint, vector_json) VALUES ('a', 'sha-a', ?, '[1.0, 0.0, 0.0]')", (FINGERPRINT,))
    conn.commit()
    assert dense_candidates(conn, [1.0, 0.0, 0.0], FINGERPRINT, KINDS) == []
    assert list(stored_vector(conn, "a", "sha-a", FINGERPRINT)) == pytest.approx([1.0, 0.0, 0.0])
    report = embed_recall_documents(conn, RecordingEmbedder(), include_raw_transcripts=False)
    assert report.embedded == 0 and report.error


def test_migration_adds_vector_columns_and_backs_up_the_legacy_database_once(tmp_path):
    from dream import open_db
    path = tmp_path / "dream.db"
    conn = open_db(path)
    conn.execute("ALTER TABLE recall_embeddings DROP COLUMN vector_blob")
    conn.execute("ALTER TABLE recall_embeddings DROP COLUMN dimension")
    conn.commit()
    conn.close()
    conn = open_db(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(recall_embeddings)")}
    assert {"vector_blob", "dimension", "vector_json"} <= columns
    assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 4
    conn.close()
    backups = sorted(tmp_path.glob("dream.db.bak-*"))
    assert len(backups) == 1
    open_db(path).close()
    assert sorted(tmp_path.glob("dream.db.bak-*")) == backups


def test_dense_scan_p95_over_5000_vectors_of_1024_dimensions(tmp_path):
    conn = _db(tmp_path)
    rng = random.Random(7)
    dimension = 1024
    rows = []
    for index in range(5000):
        doc_id = f"doc-{index:05d}"
        rows.append((doc_id, f"sha-{index}", "approved_memory", "user_approved", None, f"{doc_id}.md", "", "", "v1", "x"))
    conn.executemany("INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, source_updated_at, indexed_at, source_version, text) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    for index in range(5000):
        store_vector(conn, f"doc-{index:05d}", f"sha-{index}", FINGERPRINT, [rng.uniform(-1.0, 1.0) for _ in range(dimension)])
    conn.commit()
    query = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    timings = []
    for _ in range(8):
        started = time.perf_counter()
        results = dense_candidates(conn, query, FINGERPRINT, KINDS, limit=200)
        timings.append((time.perf_counter() - started) * 1000)
    assert len(results) == 200
    assert all(left[1] >= right[1] for left, right in zip(results, results[1:]))
    timings.sort()
    p95 = timings[max(0, math.ceil(len(timings) * 0.95) - 1)]
    assert p95 < 400, f"dense scan p95 {p95:.0f} ms"


# --- nightly embedding -------------------------------------------------------


def test_embedding_sends_batches_of_32_and_skips_documents_that_have_vectors(tmp_path):
    conn = _db(tmp_path)
    for index in range(70):
        _doc(conn, f"d{index:02d}")
    conn.commit()
    embedder = RecordingEmbedder()
    report = embed_recall_documents(conn, embedder, include_raw_transcripts=False)
    assert [len(batch) for batch in embedder.batches] == [32, 32, 6]
    assert (report.candidates, report.embedded, report.error) == (70, 70, None)
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings WHERE vector_blob IS NOT NULL AND dimension = 3 AND vector_json = '[]'").fetchone()[0] == 70
    again = RecordingEmbedder()
    assert embed_recall_documents(conn, again, include_raw_transcripts=False).candidates == 0
    assert again.batches == []
    refresh = RecordingEmbedder()
    assert embed_recall_documents(conn, refresh, include_raw_transcripts=False, refresh=True).embedded == 70
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 70


def test_embedding_failure_keeps_completed_batches_and_reports_once(tmp_path):
    conn = _db(tmp_path)
    for index in range(40):
        _doc(conn, f"d{index:02d}")
    conn.commit()
    report = embed_recall_documents(conn, RecordingEmbedder(fail_after=1), include_raw_transcripts=False)
    assert (report.candidates, report.embedded) == (40, 32)
    assert report.error == "RuntimeError: offline"
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 32
    assert conn.execute("SELECT COUNT(*) FROM recall_documents").fetchone()[0] == 40


def test_embedding_rejects_a_wrong_vector_count_without_writing(tmp_path):
    class ShortEmbedder(RecordingEmbedder):
        def embed(self, texts):
            return super().embed(texts)[:-1]
    conn = _db(tmp_path)
    _doc(conn, "a")
    _doc(conn, "b")
    conn.commit()
    report = embed_recall_documents(conn, ShortEmbedder(), include_raw_transcripts=False)
    assert report.embedded == 0 and "count" in report.error
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 0


def test_embedding_sends_scrubbed_text(tmp_path):
    conn = _db(tmp_path)
    _doc(conn, "a", text="GITHUB_TOKEN=ghp_abcdefghijklmnop and the release notes")
    conn.commit()
    embedder = RecordingEmbedder()
    embed_recall_documents(conn, embedder, include_raw_transcripts=False)
    (sent,) = embedder.batches[0]
    assert "ghp_abcdefghijklmnop" not in sent and "release notes" in sent


def test_sync_removes_vectors_of_deleted_and_changed_documents(tmp_path):
    from dream import open_db
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "a.md").write_text("alpha", encoding="utf-8")
    (memory / "b.md").write_text("beta", encoding="utf-8")
    conn = open_db(tmp_path / "dream.db")
    synchronize_recall_documents(conn, memory, include_raw_transcripts=False, constitution_path=None)
    embed_recall_documents(conn, RecordingEmbedder(), include_raw_transcripts=False)
    store_vector(conn, "unrelated", "sha", "other:fp:3", [1.0, 0.0, 0.0])
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 3
    (memory / "b.md").unlink()
    (memory / "a.md").write_text("alpha, revised", encoding="utf-8")
    synchronize_recall_documents(conn, memory, include_raw_transcripts=False, constitution_path=None)
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 0
    report = embed_recall_documents(conn, RecordingEmbedder(), include_raw_transcripts=False)
    assert report.embedded == 1
