from recall_adapters import apply_optional_adapters
from recall_types import AdapterSettings, RecallCandidate, RecallDocument, RecallQuery, RecallSettings


class FakeEmbedder:
    fingerprint = "fake-v1"
    remote_data_egress = False
    def embed(self, texts):
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
        values, mode, fallback = apply_optional_adapters(
            readonly,
            query,
            [_candidate()],
            _settings(embed=True),
            embedder=FakeEmbedder(),
            allow_cache_writes=False,
        )
        assert values and mode == "lexical_plus_embedder" and fallback is None
        assert readonly.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone() == (0,)
        readonly.close()
        assert sorted(path.name for path in db_dir.iterdir()) == ["dream.db"]
    finally:
        db_dir.chmod(0o755)
