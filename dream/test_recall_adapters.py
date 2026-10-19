import pytest

from recall_adapters import apply_optional_adapters
from recall_types import AdapterSettings, RecallCandidate, RecallDocument, RecallQuery, RecallSettings


class FakeEmbedder:
    fingerprint = "fake-v1"
    remote_data_egress = False
    def __init__(self):
        self.calls = []
    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class BrokenReranker:
    fingerprint = "broken-v1"
    remote_data_egress = False
    def score(self, query, texts):
        raise RuntimeError("offline")


def _settings(embed=False, rerank=False):
    return RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(embed, "fake", False), AdapterSettings(rerank, "fake", False))


def _candidate():
    doc = RecallDocument("a", "sha", "approved_memory", "user_approved", None, "a.md", "", "", "v1", "text")
    return RecallCandidate(doc, "text", {"fts": 1}, .2, 1)


def test_embedder_is_cached_and_broken_reranker_falls_back(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    query = RecallQuery("query", "s", "prompt", None, (), 4000, frozenset(), False)
    values, mode, fallback = apply_optional_adapters(conn, query, [_candidate()], _settings(embed=True, rerank=True), embedder=FakeEmbedder(), reranker=BrokenReranker())
    assert values and mode == "lexical_plus_embedder"
    assert fallback and "reranker-error" in fallback
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 1


def test_unavailable_adapter_keeps_lexical_candidates(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    query = RecallQuery("query", "s", "prompt", None, (), 4000, frozenset(), False)
    values, mode, fallback = apply_optional_adapters(conn, query, [_candidate()], _settings(embed=True), embedder=None)
    assert values[0].document.id == "a"
    assert mode == "lexical" and fallback == "embedder-unavailable"


def test_readonly_adapter_path_does_not_write_embedding_cache(tmp_path):
    from dream import open_db, open_db_readonly

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    conn = open_db(db_path)
    conn.close()
    db_dir.chmod(0o555)
    try:
        readonly = open_db_readonly(db_path)
        query = RecallQuery("query", "s", "prompt", None, (), 4000, frozenset(), False)
        embedder = FakeEmbedder()
        values, mode, fallback = apply_optional_adapters(
            readonly,
            query,
            [_candidate()],
            _settings(embed=True),
            embedder=embedder,
            allow_cache_writes=False,
        )
        assert embedder.calls == [["query"]]
        # Read-only paths never embed documents: only the query was sent, the
        # candidate without a stored vector keeps its lexical score, and the
        # mode stays lexical because nothing was rescored.
        assert values == [_candidate()] and mode == "lexical" and fallback is None
        assert readonly.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone() == (0,)
        readonly.close()
        assert sorted(path.name for path in db_dir.iterdir()) == ["dream.db"]
    finally:
        db_dir.chmod(0o755)


def test_cached_embedding_is_stored_as_a_float32_blob(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    query = RecallQuery("query", "s", "prompt", None, (), 4000, frozenset(), False)
    apply_optional_adapters(conn, query, [_candidate()], _settings(embed=True), embedder=FakeEmbedder())
    row = conn.execute("SELECT vector_json, length(vector_blob), dimension FROM recall_embeddings").fetchone()
    assert row == ("[]", 8, 2)


def test_stored_blob_rescores_on_the_read_only_path_with_a_supplied_query_vector(tmp_path):
    from dream import open_db
    from recall_embeddings import store_vector
    conn = open_db(tmp_path / "dream.db")
    store_vector(conn, "a", "sha", "fake-v1", [1.0, 0.0])
    conn.commit()
    embedder = FakeEmbedder()
    query = RecallQuery("query", "s", "prompt", None, (), 4000, frozenset(), False)
    values, mode, fallback = apply_optional_adapters(
        conn, query, [_candidate()], _settings(embed=True), embedder=embedder,
        allow_cache_writes=False, query_vector=(1.0, 0.0),
    )
    assert embedder.calls == []
    assert mode == "lexical_plus_embedder" and fallback is None
    assert values[0].score == pytest.approx((0.2 + 1.0) / 2)


def test_unreachable_embedder_reports_unavailable_during_rescore(tmp_path):
    from dream import open_db
    from recall_adapters import EmbedderUnavailable

    class Down(FakeEmbedder):
        def embed(self, texts):
            raise EmbedderUnavailable("refused")

    conn = open_db(tmp_path / "dream.db")
    query = RecallQuery("query", "s", "prompt", None, (), 4000, frozenset(), False)
    values, mode, fallback = apply_optional_adapters(conn, query, [_candidate()], _settings(embed=True), embedder=Down())
    assert values == [_candidate()] and mode == "lexical" and fallback == "embedder-unavailable"
