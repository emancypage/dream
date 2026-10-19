import sqlite3
from pathlib import Path

from recall_query import build_safe_fts_match, extract_structured_terms, normalize_path_text, normalize_query_text, rank_lexical
from recall_types import RecallQuery


def _db(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    rows = [
        ("a", "approved_memory", "user_approved", "-home-a-api", "api.md", "postgres migration runbook"),
        ("b", "distilled_summary", "model_distilled", "-home-b-web", "web.md", "postgres deployment summary"),
    ]
    for i, (doc_id, kind, trust, project, path, text) in enumerate(rows):
        conn.execute(
            "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, source_updated_at, indexed_at, source_version, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, doc_id, kind, trust, project, path, f"2026-08-20T10:0{i}:00Z", "2026-08-20T10:00:00Z", "v1", text),
        )
    conn.commit()
    return conn


def test_normalization_and_structured_terms_are_deterministic():
    assert normalize_query_text("  Résumé\\API  ") == "résumé/api"
    assert normalize_path_text("/Home/A/API\\src") == "home a api src"
    assert extract_structured_terms("Fix PROJ-123 with git and PROJ-123") == ("proj-123", "git")


def test_fts_match_quotes_user_syntax():
    match = build_safe_fts_match('alpha OR beta) NEAR("secret")')
    assert match is not None
    assert match == '"alpha" OR "beta" OR "near" OR "secret"'
    assert ")" not in match
    assert '"alpha"' in match


def test_lexical_rank_has_component_ranks_and_soft_cross_project_results(tmp_path):
    conn = _db(tmp_path)
    query = RecallQuery("postgres", "s1", "prompt", "/home/a/api", ("/home/a/api",), 4000, frozenset(), False)
    results = rank_lexical(conn, query)
    assert results
    assert results[0].document.id == "a"
    assert results[0].component_ranks
    assert {candidate.document.id for candidate in results} == {"a", "b"}


def test_natural_language_prompt_finds_topic_without_matching_every_word(tmp_path):
    conn = _db(tmp_path)
    query = RecallQuery("How can I improve the postgres migration?", "s1", "prompt", "/home/a/api", (), 4000, frozenset(), False)
    results = rank_lexical(conn, query)
    assert results[0].document.id == "a"
    assert "fts" in results[0].component_ranks


def test_prompt_does_not_recall_documents_only_because_of_working_directory(tmp_path):
    conn = _db(tmp_path)
    query = RecallQuery("pottery glazing", "s1", "prompt", "/home/a/api", (), 4000, frozenset(), False)
    assert rank_lexical(conn, query) == []


def test_home_directory_alone_does_not_recall_other_projects(tmp_path):
    conn = _db(tmp_path)
    query = RecallQuery(str(Path.home()), "s1", "session-start:startup", str(Path.home()), (), 6000, frozenset(), False)
    conn.execute("UPDATE recall_documents SET project_slug = ?", (str(Path.home()).replace("/", "-") + "-unrelated",))
    conn.commit()
    assert rank_lexical(conn, query) == []


def _signature(results):
    return [(c.document.id, dict(c.component_ranks), c.score, c.best_component_rank, c.scrubbed_excerpt) for c in results]


def test_dense_none_and_empty_dense_are_identical_to_lexical(tmp_path):
    conn = _db(tmp_path)
    for query in (
        RecallQuery("postgres migration", "s1", "prompt", "/home/a/api", ("/home/a/api",), 4000, frozenset(), False),
        RecallQuery("/home/a/api", "s1", "session-start:startup", "/home/a/api", ("/home/a/api",), 6000, frozenset(), False),
    ):
        baseline = _signature(rank_lexical(conn, query))
        assert baseline
        assert _signature(rank_lexical(conn, query, dense=None)) == baseline
        assert _signature(rank_lexical(conn, query, dense=[])) == baseline


def test_dense_only_document_enters_prompt_candidates_as_fifth_list(tmp_path):
    conn = _db(tmp_path)
    query = RecallQuery("pottery glazing", "s1", "prompt", None, (), 4000, frozenset(), False)
    assert rank_lexical(conn, query) == []
    results = rank_lexical(conn, query, dense=[("b", 0.91), ("a", 0.42), ("missing", 0.99)])
    assert [candidate.document.id for candidate in results] == ["b", "a"]
    assert results[0].component_ranks["dense"] == 1
    assert "fts" not in results[0].component_ranks
    assert results[0].scrubbed_excerpt == "postgres deployment summary"


def test_dense_list_fuses_with_lexical_lists_by_score_on_the_prompt_path(tmp_path):
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, source_updated_at, indexed_at, source_version, text) "
        "VALUES ('c', 'c', 'approved_memory', 'user_approved', NULL, 'db.md', '2026-08-20T10:05:00Z', '2026-08-20T10:00:00Z', 'v1', 'database choice notes')")
    conn.commit()
    query = RecallQuery("postgres", "s1", "prompt", None, (), 4000, frozenset(), False)
    lexical = rank_lexical(conn, query)
    assert {candidate.document.id for candidate in lexical} == {"a", "b"}
    fused = rank_lexical(conn, query, dense=[("c", 0.9), ("b", 0.8)])
    ids = [candidate.document.id for candidate in fused]
    assert set(ids) == {"a", "b", "c"}
    # b has both FTS and dense evidence, so it outranks FTS-only a; fused order is by score.
    assert ids[0] == "b"
    assert [candidate.score for candidate in fused] == sorted((candidate.score for candidate in fused), reverse=True)


def test_dense_ids_obey_exclusions_and_the_raw_transcript_rule(tmp_path):
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, source_updated_at, indexed_at, source_version, text) "
        "VALUES ('raw', 'raw', 'raw_transcript', 'untrusted_transcript', NULL, 'transcript:x', '', '', 'v1', 'raw session text')")
    conn.commit()
    query = RecallQuery("pottery", "s1", "prompt", None, (), 4000, frozenset({"a"}), False)
    results = rank_lexical(conn, query, dense=[("a", 0.9), ("raw", 0.8), ("b", 0.7)])
    assert [candidate.document.id for candidate in results] == ["b"]
