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


# ------------------------------------------------------------ usage tie-break


def _tie_db(tmp_path):
    """Two approved documents whose fused scores tie exactly at session start.

    x: paths rank 1, recency rank 2; y: paths rank 2, recency rank 1. Their
    scores are the same two terms summed, trust is equal, and today the newer
    y wins the tie. z is unrelated to the working directory.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    rows = [
        ("x", "-proj-api-web", "x.md", "deployment notes", "2026-08-20T10:00:00Z"),
        ("y", "-proj-api", "y.md", "release checklist", "2026-08-21T10:00:00Z"),
        ("z", "-other-thing", "z.md", "pottery glazing", "2026-08-22T10:00:00Z"),
    ]
    for doc_id, project, path, text, updated in rows:
        conn.execute(
            "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, "
            "source_updated_at, indexed_at, source_version, text) VALUES (?, ?, 'approved_memory', 'user_approved', "
            "?, ?, ?, '2026-08-20T10:00:00Z', 'v1', ?)",
            (doc_id, doc_id, project, path, updated, text),
        )
    conn.commit()
    return conn


def _session_start(cwd="/home/proj/api/web"):
    return RecallQuery(cwd, "s1", "session-start:startup", cwd, (), 6000, frozenset(), False)


def test_usage_breaks_an_exact_tie_after_trust(tmp_path):
    conn = _tie_db(tmp_path)
    baseline = rank_lexical(conn, _session_start())
    assert [c.document.id for c in baseline] == ["y", "x"]
    assert baseline[0].score == baseline[1].score  # the premise: an exact tie
    assert baseline[0].best_component_rank == baseline[1].best_component_rank == 1

    used = rank_lexical(conn, _session_start(), usage={"x": (5, "2026-10-01T00:00:00Z"), "y": (2, "2026-10-01T00:00:00Z")})
    assert [c.document.id for c in used] == ["x", "y"]
    assert [c.score for c in used] == [c.score for c in reversed(baseline)]


def test_usage_never_adds_a_candidate_or_changes_order_without_a_tie(tmp_path):
    conn = _tie_db(tmp_path)
    heavy = {"z": (1000, "2026-10-19T00:00:00Z")}
    assert "z" not in {c.document.id for c in rank_lexical(conn, _session_start(), usage=heavy)}
    prompt = RecallQuery("deployment release", "s1", "prompt", "/home/proj/api/web", (), 4000, frozenset(), False)
    off = [c.document.id for c in rank_lexical(conn, prompt)]
    on = [c.document.id for c in rank_lexical(conn, prompt, usage={"y": (1000, "x"), "z": (1000, "x")})]
    assert on == off and "z" not in on


def test_rankings_without_usage_are_unchanged(tmp_path):
    conn = _tie_db(tmp_path)
    for query in (_session_start(), RecallQuery("deployment", "s1", "prompt", None, (), 4000, frozenset(), False)):
        plain = rank_lexical(conn, query)
        assert rank_lexical(conn, query, usage=None) == plain
        assert rank_lexical(conn, query, usage={}) == plain


def test_usage_only_reorders_within_exact_ties(tmp_path):
    """On a random store, usage permutes candidates only inside groups whose
    ranking evidence before the tie-break (FTS rank on prompts, score, trust) is identical."""
    import random
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dream import open_db
    rng = random.Random(20261020)
    tokens = ["api", "web", "proj", "core", "docs", "infra", "mobile"]
    trusts = [("approved_memory", "user_approved"), ("distilled_summary", "model_distilled")]
    conn = open_db(tmp_path / "dream.db")
    for i in range(40):
        kind, trust = rng.choice(trusts)
        slug = "-" + "-".join(rng.sample(tokens, rng.randint(1, 3)))
        text = " ".join(rng.sample(["deploy", "release", "migration", "runbook", "notes", "alpha"], 2))
        conn.execute(
            "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, "
            "source_updated_at, indexed_at, source_version, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'v1', ?)",
            (f"d{i:02d}", f"d{i:02d}", kind, trust, slug, f"f{i:02d}.md",
             f"2026-08-{rng.randint(1, 3):02d}T10:00:00Z", "2026-08-20T10:00:00Z", text),
        )
    # Same path evidence with timestamps rising by ID: the paths order is the
    # reverse of the recency order, so the k-th and (n+1-k)-th documents tie.
    for i in range(20):
        kind, trust = rng.choice(trusts)
        conn.execute(
            "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, "
            "source_updated_at, indexed_at, source_version, text) VALUES (?, ?, ?, ?, '-tiezone', ?, ?, "
            "'2026-08-20T10:00:00Z', 'v1', 'shared notes')",
            (f"t{i:02d}", f"t{i:02d}", kind, trust, f"t{i:02d}.md", f"2026-08-05T10:00:{i:02d}Z"),
        )
    conn.commit()
    usage = {f"{prefix}{i:02d}": (rng.randint(0, 50), "2026-10-01T00:00:00Z") for prefix in "dt" for i in range(40)}
    trust_order = {"user_approved": 0, "model_distilled": 1}
    queries = [_session_start("/home/proj/api/web"), _session_start("/srv/core/docs/infra"), _session_start("/srv/tiezone"),
               RecallQuery("deploy release notes", "s1", "prompt", "/home/proj/api", (), 4000, frozenset(), False)]
    tied_groups = 0
    for query in queries:
        is_prompt = query.hook_event == "prompt"

        def evidence(candidate):
            return (candidate.component_ranks.get("fts", 0) if is_prompt else 0, -candidate.score,
                    trust_order[candidate.document.trust_level])

        off = rank_lexical(conn, query)
        on = rank_lexical(conn, query, usage=usage)
        assert off and {c.document.id for c in on} == {c.document.id for c in off}
        assert [evidence(c) for c in on] == [evidence(c) for c in off]
        for i in range(1, len(on)):
            if evidence(on[i - 1]) == evidence(on[i]):
                tied_groups += 1
                assert usage[on[i - 1].document.id][0] >= usage[on[i].document.id][0]
    assert tied_groups  # the store really has ties for the switch to act on
