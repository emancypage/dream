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
    assert " OR " not in match
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
