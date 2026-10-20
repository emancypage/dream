import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path


import pytest

from recall_query import RECENT_POOL_SIZE, build_safe_fts_match, extract_structured_terms, normalize_path_text, normalize_query_text, rank_lexical
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


# ------------------------------------------------------------ golden outputs and latency
#
# fixtures/recall/lexical-golden.json was captured from rank_lexical and
# run_context at bdb047d, before the
# lexical path was bounded: every query of
# the public fixture and of a 300-document synthetic store (both below
# RECENT_POOL_SIZE, where the bound changes nothing), and the prompt-path
# queries of the 10,000-document store, whose candidate pool the bound leaves
# as it was. Session starts on the large store match paths over a bounded
# pool by design (test_session_start_pool_is_bounded_on_a_large_store); their
# bounded output was captured at d33394b in lexical-golden-bounded.json. The
# four prompt-excluded embedder records were updated when session exclusions
# moved before the dense-candidate limit: the excluded documents no longer take
# dense slots, so one more eligible candidate enters. Set
# DREAM_UPDATE_GOLDEN=1 only for an intentional, reviewed ranking change.
#
# Records keep IDs, component ranks, counts and hashes exact but round every
# score to 12 decimal places (GOLDEN_SCORE_DECIMALS). Python 3.12 made sum() of
# floats compensated while 3.11 adds left to right, so the same RRF score can
# differ in its last bit between versions; 12 places keeps about 10
# significant digits of a score near 0.01, far above that noise.

GOLDEN = Path(__file__).parent / "fixtures" / "recall" / "lexical-golden.json"
BOUNDED_GOLDEN = Path(__file__).parent / "fixtures" / "recall" / "lexical-golden-bounded.json"
GOLDEN_SCORE_DECIMALS = 12
PUBLIC_FIXTURE = Path(__file__).parent / "fixtures" / "recall" / "public.json"
GOLDEN_HOME = "/home/golden"
_INDEXED_AT = "2026-10-20T00:00:00+00:00"

_PROJECTS = (
    "api", "web", "infra", "mobile", "billing", "search", "dream", "payments", "auth", "gateway",
    "analytics", "docs", "design", "ledger", "portal", "scheduler", "notifier", "importer", "exporter", "crm",
    "inventory", "shipping", "catalog", "pricing", "reports", "admin", "chat", "media", "maps", "sync",
    "vault", "metrics", "alerts", "backup", "deploy", "kiosk", "booking", "survey", "wiki", "forms",
)
_VOCABULARY = (
    "migration postgres schema index query cache redis queue worker retry timeout deploy release rollback "
    "staging production review pull request branch commit merge test fixture regression coverage lint "
    "format config secret token login session cookie header proxy ingress certificate dns latency budget "
    "memory cpu disk volume backup restore snapshot cron timer service daemon socket port loopback "
    "embedding vector recall ranking threshold calibration eval fixture language polish english decision "
    "preference project user fact note summary runbook checklist incident outage alert dashboard metric "
    "grafana prometheus kubernetes container image build pipeline artifact package dependency upgrade "
    "python typescript react django flask fastapi celery sqlalchemy alembic pydantic pytest playwright "
    "baza danych migracja wdrożenie testy przegląd gałąź zmiana konfiguracja sekret użytkownik projekt "
    "decyzja preferencja notatka podsumowanie błąd poprawka wydajność pamięć dysk kopia zapasowa "
    "harmonogram usługa serwer klient zapytanie indeks ranking próg kalibracja język polski angielski"
).split()
_SYNTHETIC_COMMANDS = ("git", "pytest", "docker", "kubectl", "npm", "make", "ssh", "curl", "uv", "terraform")
_SYNTHETIC_SECRETS = (
    "API_KEY=sk-live-{n:06d}abcdef", "postgres://admin:hunter{n}@db.internal/app", "Bearer tok{n:08d}xyz",
    "password: pw{n:05d}!", "GITHUB_TOKEN=ghp_{n:010d}",
)


class _Lcg:
    """A 64-bit LCG: the same stream on every Python version, unlike random's helpers."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def below(self, n: int) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.state >> 33) % n

    def pick(self, values):
        return values[self.below(len(values))]


def _sentence(rng: _Lcg, project: str) -> str:
    words = [rng.pick(_VOCABULARY) for _ in range(6 + rng.below(10))]
    roll = rng.below(20)
    if roll == 0:
        words.append(f"{project.upper()[:4]}-{100 + rng.below(900)}")
    elif roll == 1:
        words.append(rng.pick(_SYNTHETIC_COMMANDS))
    elif roll == 2:
        words.append(f"{GOLDEN_HOME}/dev/{project}/src/{rng.pick(_VOCABULARY)}.py")
    elif roll == 3:
        words.append(rng.pick(_SYNTHETIC_SECRETS).format(n=rng.below(10 ** 6)))
    elif roll in (4, 5, 6):
        words.append(project)
    return " ".join(words).capitalize() + "."


def _timestamp(rng: _Lcg) -> str:
    day = rng.below(640)
    year, day = 2025 + day // 365, day % 365
    month = 0
    for length in (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31):
        if day < length:
            break
        day -= length
        month += 1
    return f"{year:04d}-{month + 1:02d}-{day + 1:02d}T{rng.below(24):02d}:{rng.below(60):02d}:{rng.below(60):02d}Z"


def synthetic_rows(count: int, *, seed: int = 20261020) -> list[dict]:
    """Recall rows shaped like a live store: about 1.5% approved memory, 8.5% raw transcripts, the rest distilled.

    Distilled and transcript slugs share the ``-home-golden-dev-`` prefix, as
    the slugs of one user's projects do, and timestamps spread over 21 months.
    """
    rng = _Lcg(seed)
    slug_prefix = "-" + GOLDEN_HOME.strip("/").replace("/", "-") + "-dev-"
    rows = []
    for index in range(count):
        project = rng.pick(_PROJECTS)
        roll = rng.below(200)
        if roll < 3:
            kind, trust, slug = "approved_memory", "user_approved", None
            path = f"{rng.pick(('project', 'feedback', 'reference', 'user'))}_{project}_{index}.md"
            text = f"---\nname: {project}\n---\n\n" + "\n".join(_sentence(rng, project) for _ in range(3 + rng.below(8)))
        elif roll < 20:
            kind, trust, slug = "raw_transcript", "untrusted_transcript", slug_prefix + project
            path = f"transcript:codex:sess-{index:05d}"
            text = "\n".join(f"--- {rng.pick(('USER', 'ASSISTANT'))} ---\n" + _sentence(rng, project)
                             for _ in range(4 + rng.below(12)))
        else:
            kind, trust, slug = "distilled_summary", "model_distilled", slug_prefix + project
            path = f"distilled:codex:sess-{index:05d}"
            notes = {key: [_sentence(rng, project) for _ in range(rng.below(5))]
                     for key in ("decisions", "preferences", "projects", "user_facts")}
            text = json.dumps(notes, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows.append({
            "id": f"doc-{index:05d}", "content_sha256": digest, "source_kind": kind, "trust_level": trust,
            "project_slug": slug, "source_path": path, "source_updated_at": _timestamp(rng),
            "source_version": digest[:12], "text": text,
        })
    return rows


def _public_rows() -> list[dict]:
    rows = []
    for document in json.loads(PUBLIC_FIXTURE.read_text(encoding="utf-8"))["documents"]:
        text = document["text"]
        rows.append({**document, "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text})
    return rows


class _WordEmbedder:
    """Deterministic 8-dimensional bag-of-word-sums vectors, enough to exercise the dense list."""

    fingerprint = "golden-words:8"
    remote_data_egress = False
    server_model = None

    def embed(self, texts):
        return [_word_vector(text) for text in texts]


def _word_vector(text: str) -> list[float]:
    values = [0.5] + [0.0] * 7
    for word in re.findall(r"\w+", text.casefold()):
        values[sum(map(ord, word)) % 8] += 1.0
    return values


def _build_store(path: Path, rows: list[dict]):
    """A migrated store written by the sync writer, with vectors, usage and both calibrations."""
    from dream import open_db
    from recall_documents import _upsert_document
    from recall_embeddings import store_vector
    from recall_select import embedder_calibration_version

    conn = open_db(path)
    with conn:
        for row in rows:
            _upsert_document(conn, row, _INDEXED_AT)
        for row in rows:
            store_vector(conn, row["id"], row["content_sha256"], _WordEmbedder.fingerprint, _word_vector(row["text"]))
        for index, row in enumerate(rows):
            if index % 9 == 1:
                conn.execute("INSERT INTO recall_usage VALUES (?, '2026-09-01T00:00:00Z', '2026-10-01T00:00:00Z', ?)",
                             (row["id"], 1 + index % 13))
        conn.execute("INSERT INTO recall_calibrations(mode, calibration_version, threshold, fixture_sha256) "
                     "VALUES ('lexical', 'golden-lexical', 0.012, 'golden')")
        conn.execute("INSERT INTO recall_calibrations(mode, calibration_version, threshold, fixture_sha256) "
                     "VALUES ('lexical_plus_embedder', ?, 0.004, 'golden')",
                     (embedder_calibration_version("golden", "lexical_plus_embedder", _WordEmbedder.fingerprint),))
    return conn


def _golden_queries(corpus: str) -> dict[str, RecallQuery]:
    def start(cwd, roots=(), *, source="startup", excluded=(), raw=False):
        return RecallQuery(" ".join([cwd, *roots]), "s1", f"session-start:{source}", cwd, tuple(roots), 18000,
                           frozenset(excluded), raw)

    def prompt(text, cwd, *, event="prompt", roots=(), excluded=(), raw=False):
        return RecallQuery(text, "s1", event, cwd, tuple(roots), 4000, frozenset(excluded), raw)

    if corpus == "public":
        data = json.loads(PUBLIC_FIXTURE.read_text(encoding="utf-8"))
        queries = {item["id"]: prompt(item["query"], item.get("cwd"), roots=item.get("repository_roots", ()))
                   for item in data["queries"]}
        queries["start-api"] = start("/home/a/api", ("/home/a/api",))
        queries["start-web"] = start("/home/b/web")
        queries["start-pytest"] = start("/srv/pytest/api")
        return queries
    billing, dream_dir = f"{GOLDEN_HOME}/dev/billing", f"{GOLDEN_HOME}/dev/dream"
    return {
        "start-billing": start(billing, (billing,)),
        "start-dream": start(dream_dir, (dream_dir,)),
        "start-home": start(GOLDEN_HOME),
        "start-resume-excluded": start(billing, source="resume", excluded=("doc-00007", "doc-00042", "doc-00120")),
        "start-raw": start(f"{GOLDEN_HOME}/dev/vault", raw=True),
        "prompt-en": prompt("How do we run the postgres migration with pytest for billing BILL-123?", billing),
        "prompt-pl": prompt("jak zrobić migracja bazy danych w projekcie billing", billing),
        "prompt-nomatch": prompt("pottery glazing kiln", billing),
        "prompt-raw": prompt("docker kubectl deploy rollback", billing, raw=True),
        "prompt-excluded": prompt("grafana alert dashboard", billing, excluded=("doc-00007", "doc-00042")),
        "agent-query": prompt("release checklist decision", billing, event="agent-query", roots=("billing",)),
        "consolidate-subset": prompt("staging release checklist decision", None, event="consolidate-subset"),
    }


def _golden_dense(rows: list[dict]) -> list[tuple[str, float]]:
    rng = _Lcg(7)
    ids = list(dict.fromkeys(rows[rng.below(len(rows))]["id"] for _ in range(40)))
    ids.insert(min(3, len(ids)), "doc-missing")
    return [(doc_id, round(0.95 - 0.01 * rank, 6)) for rank, doc_id in enumerate(ids)]


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ranking_record(results) -> dict:
    entries = [
        [c.document.id, [list(item) for item in c.component_ranks.items()], round(c.score, GOLDEN_SCORE_DECIMALS),
         c.best_component_rank,
         hashlib.sha256(c.scrubbed_excerpt.encode("utf-8")).hexdigest()[:16]]
        for c in results
    ]
    return {"count": len(entries), "top": [entry[:4] for entry in entries[:3]], "sha256": _digest(entries)}


def _golden_settings(*, embedder: bool, usage: bool):
    from recall_types import AdapterSettings, RecallSettings

    return RecallSettings(True, False, False, True, 18000, 4000, 18000, 1200, "", "",
                          AdapterSettings(embedder, "golden", False), AdapterSettings(False, "none", False),
                          usage_tie_break=usage)


def _context_record(result) -> dict:
    selected = [c.document.id for c in result.selected]
    return {
        "selected_count": len(selected), "selected_top": selected[:5], "selected_sha256": _digest(selected),
        "mode": result.mode,
        "calibration_version": result.calibration_version,
        "fallback_reason": result.diagnostics.fallback_reason,
        "candidate_count": result.diagnostics.candidate_count,
        "context_sha256": hashlib.sha256(result.rendered_context.encode("utf-8")).hexdigest(),
    }


def _golden_records(conn, rows, corpus: str, tmp_path: Path, *, path: str | None = None,
                    contexts: bool = True) -> dict:
    """Golden records of every query, or only of the ``"prompt"`` or ``"session-start"`` path."""
    from recall_context import run_context
    from recall_usage import usage_map

    usage = usage_map(conn)
    assert usage  # the store has usage rows for the switch to act on
    dense = _golden_dense(rows)
    records = {}
    for name, query in _golden_queries(corpus).items():
        if path is not None and query.hook_event.startswith("session-start") != (path == "session-start"):
            continue
        for dense_name, dense_list in (("lexical", None), ("dense", dense)):
            for usage_name, usage_value in (("plain", None), ("usage", usage)):
                records[f"{corpus}/{name}/rank/{dense_name}/{usage_name}"] = _ranking_record(
                    rank_lexical(conn, query, dense=dense_list, usage=usage_value))
        if not contexts:
            continue
        for embed in (False, True):
            for tie_break in (False, True):
                result = run_context(conn, query, _golden_settings(embedder=embed, usage=tie_break),
                                     embedder=_WordEmbedder() if embed else None,
                                     memory_root=tmp_path / "memory", constitution_path=tmp_path / "constitution.md")
                key = f"{corpus}/{name}/context/{'embedder' if embed else 'lexical'}/{'usage' if tie_break else 'plain'}"
                records[key] = _context_record(result)
    return records


@pytest.fixture(scope="module")
def large_store(tmp_path_factory):
    """The 10,000-document synthetic store, built once by the sync writer and opened read-only: ``(conn, rows)``."""
    from storage import open_db_readonly

    rows = synthetic_rows(10_000)
    path = tmp_path_factory.mktemp("synthetic-10k") / "dream.db"
    _build_store(path, rows).close()
    conn = open_db_readonly(path)
    yield conn, rows
    conn.close()


def _golden_text(golden: dict) -> str:
    """One record per line, so a ranking change shows up as a readable diff."""
    lines = [f" {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}" for key, value in sorted(golden.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _assert_matches_golden(records: dict, corpus: str, *, ranks_only: bool = False, golden_path: Path = GOLDEN) -> None:
    golden = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.exists() else {}
    if os.environ.get("DREAM_UPDATE_GOLDEN") == "1" and not ranks_only:
        golden = {key: value for key, value in golden.items() if not key.startswith(f"{corpus}/")}
        golden.update(records)
        golden_path.write_text(_golden_text(golden), encoding="utf-8")
    expected = {key: value for key, value in golden.items()
                if key.startswith(f"{corpus}/") and (not ranks_only or "/rank/" in key)}
    assert expected, "golden file has no records for this corpus"
    mismatched = [key for key in sorted(expected) if records.get(key) != expected[key]]
    assert not mismatched, (f"{len(mismatched)} records differ, first {mismatched[0]}: "
                            f"{records.get(mismatched[0])} != {expected[mismatched[0]]}")
    assert set(records) == set(expected)


@pytest.mark.parametrize("corpus", ["public", "synthetic-300"])
def test_small_store_rankings_and_contexts_match_the_pre_bound_golden(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", GOLDEN_HOME)
    rows = _public_rows() if corpus == "public" else synthetic_rows(300)
    conn = _build_store(tmp_path / "dream.db", rows)
    _assert_matches_golden(_golden_records(conn, rows, corpus, tmp_path), corpus)


def test_large_store_prompt_rankings_and_contexts_match_the_pre_bound_golden(large_store, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", GOLDEN_HOME)
    conn, rows = large_store
    _assert_matches_golden(_golden_records(conn, rows, "synthetic-10k", tmp_path, path="prompt"), "synthetic-10k")


def test_large_store_session_starts_match_the_bounded_golden(large_store, tmp_path, monkeypatch):
    """The bounded session-start output, pinned so a later change to the pool or the ranking shows up."""
    monkeypatch.setenv("HOME", GOLDEN_HOME)
    conn, rows = large_store
    _assert_matches_golden(_golden_records(conn, rows, "synthetic-10k", tmp_path, path="session-start"),
                           "synthetic-10k", golden_path=BOUNDED_GOLDEN)


@pytest.mark.parametrize("shape", ["half-null", "unmigrated"])
def test_missing_path_text_ranks_like_the_golden_on_a_read_only_store(shape, tmp_path, monkeypatch):
    """Rows indexed before the column existed (NULL) and a database never
    migrated (no column at all) compute path evidence on the fly. Only
    rank_lexical reads path_text, so its records are the ones compared."""
    from storage import open_db_readonly

    monkeypatch.setenv("HOME", GOLDEN_HOME)
    rows = synthetic_rows(300)
    conn = _build_store(tmp_path / "dream.db", rows)
    if shape == "half-null":
        conn.execute("UPDATE recall_documents SET path_text = NULL WHERE rowid % 2 = 0")
    else:
        conn.execute("ALTER TABLE recall_documents DROP COLUMN path_text")
    conn.commit()
    conn.close()
    readonly = open_db_readonly(tmp_path / "dream.db")
    try:
        _assert_matches_golden(_golden_records(readonly, rows, "synthetic-300", tmp_path, contexts=False),
                               "synthetic-300", ranks_only=True)
    finally:
        readonly.close()


def test_session_start_pool_is_bounded_on_a_large_store(large_store, monkeypatch):
    """Structure and path evidence come only from FTS matches plus the most recent documents."""
    monkeypatch.setenv("HOME", GOLDEN_HOME)
    conn, rows = large_store
    query = _golden_queries("synthetic-10k")["start-billing"]
    fts_ids = {row[0] for row in conn.execute(
        "SELECT d.id FROM recall_documents_fts f JOIN recall_documents d ON d.rowid=f.rowid "
        "WHERE recall_documents_fts MATCH ? AND d.source_kind != 'raw_transcript' "
        "ORDER BY bm25(recall_documents_fts), d.id LIMIT 500", (build_safe_fts_match(query.query_text),))}
    # Rows come in ID order and the sort is stable: newest first, ties by ID, as in SQLite.
    eligible = sorted((row for row in rows if row["source_kind"] != "raw_transcript"),
                      key=lambda row: row["source_updated_at"], reverse=True)
    recent_ids = {row["id"] for row in eligible[:RECENT_POOL_SIZE]}
    billing = [row for row in eligible if (row["project_slug"] or "").endswith("-billing")]

    results = rank_lexical(conn, query)
    ids = {candidate.document.id for candidate in results}
    assert fts_ids <= ids
    assert ids <= fts_ids | recent_ids
    assert len(ids) > len(fts_ids)  # recent documents with path evidence alone still join
    # Older same-project documents without FTS evidence stay out of the pool.
    assert any(row["id"] not in fts_ids | recent_ids for row in billing)
    assert not ({row["id"] for row in billing} - fts_ids - recent_ids) & ids


def test_retention_excluded_rows_never_fill_the_session_start_pool(tmp_path):
    """A stale index (policy changed, no sync yet) still gives eligible documents the recent pool's places."""
    from config import load_config
    from recall_context import run_context
    from retention_policy import POLICY_FILE
    from dream import open_db
    memory = tmp_path / "memory"
    memory.mkdir()
    conn = open_db(tmp_path / "dream.db")
    rows = [(f"excluded-{n:04d}", "-work-secretproj", f"2026-10-19T{n // 60 % 24:02d}:{n % 60:02d}:00Z",
             f"Secret project note {n}.") for n in range(RECENT_POOL_SIZE)]
    # Older, and tied to the session only by its project slug, which full-text search does not index.
    rows.append(("allowed", "-srv-zeta", "2026-01-01T00:00:00Z", "Prefers short release notes."))
    for doc_id, project, updated, text in rows:
        conn.execute(
            "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, "
            "source_updated_at, indexed_at, source_version, text) VALUES (?, ?, 'distilled_summary', "
            "'model_distilled', ?, ?, ?, ?, 'v1', ?)",
            (doc_id, doc_id, project, f"distilled:{doc_id}", updated, updated, text),
        )
    conn.commit()
    (memory / POLICY_FILE).write_text(json.dumps({"project_slugs": ["-work-secretproj"]}), encoding="utf-8")
    query = RecallQuery("/srv/zeta", "s1", "session-start:startup", "/srv/zeta", ("/srv/zeta",), 6000,
                        frozenset(), False)
    result = run_context(conn, query, load_config(tmp_path / "missing.toml").recall_settings(),
                         memory_root=memory, constitution_path=None)
    assert [candidate.document.id for candidate in result.selected] == ["allowed"]
    conn.close()


def _second_worst_ms(run, runs: int) -> float:
    """The p95 of ``runs`` timed calls after a warm-up: the second-worst of 20."""
    run()
    run()
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        run()
        times.append((time.perf_counter() - started) * 1000)
    return sorted(times)[-2]


def test_recall_latency_budget_on_10000_documents(large_store, tmp_path, monkeypatch, assert_latency):
    """Session-start p95 <= 150 ms, prompt p95 <= 250 ms (scaled on CI), warmed read-only connection, embedder off."""
    from config import load_config
    from recall_context import run_context

    monkeypatch.setenv("HOME", GOLDEN_HOME)
    conn, _rows = large_store
    settings = load_config(tmp_path / "missing-config.toml").recall_settings()
    queries = _golden_queries("synthetic-10k")

    def timed(name):
        def run():
            return run_context(conn, queries[name], settings, memory_root=tmp_path / "memory",
                               constitution_path=tmp_path / "constitution.md")
        assert run().diagnostics.candidate_count >= 400  # real work: hundreds of candidates
        return _second_worst_ms(run, 20)

    session_start = timed("start-dream")
    prompt = timed("prompt-en")
    assert_latency("recall-session-start-10k", session_start, 150)
    assert_latency("recall-prompt-10k", prompt, 250)


# ------------------------------------------------------------ lazy scrubbing


def test_excerpts_are_scrubbed_once_and_only_when_read(tmp_path, monkeypatch):
    import recall_query
    from dataclasses import replace
    from recall_scrub import scrub_text
    from recall_types import RecallCandidate

    monkeypatch.setenv("HOME", GOLDEN_HOME)
    conn = _db(tmp_path)
    conn.execute("UPDATE recall_documents SET text = text || ' API_KEY=sk-live-123456 under /home/golden/dev'")
    conn.commit()
    calls = []

    def counting_scrub(text, home):
        calls.append(text)
        return scrub_text(text, home)

    monkeypatch.setattr(recall_query, "scrub_text", counting_scrub)
    results = rank_lexical(conn, RecallQuery("postgres", "s1", "prompt", None, (), 4000, frozenset(), False))
    assert len(results) == 2 and calls == []  # no structured term: nothing scrubbed yet

    first = results[0].scrubbed_excerpt
    assert "sk-live" not in first and "<redacted>" in first and "<home>" in first
    assert results[0].scrubbed_excerpt is first and len(calls) == 1  # kept, not scrubbed again
    plain = RecallCandidate(results[0].document, first, dict(results[0].component_ranks),
                            results[0].score, results[0].best_component_rank)
    assert results[0] == plain and plain == results[0] and results[1] != plain
    assert replace(results[1], score=1.0).scrubbed_excerpt == scrub_text(results[1].document.text, Path.home())


def test_structured_matching_scrubs_only_documents_whose_raw_text_has_a_term(tmp_path, monkeypatch):
    import recall_query
    from recall_scrub import scrub_text

    monkeypatch.setenv("HOME", GOLDEN_HOME)
    conn = _db(tmp_path)
    conn.execute("UPDATE recall_documents SET text = 'postgres runbook: run pytest first' WHERE id = 'a'")
    conn.commit()
    calls = []
    monkeypatch.setattr(recall_query, "scrub_text", lambda text, home: calls.append(text) or scrub_text(text, home))
    results = rank_lexical(conn, RecallQuery("postgres pytest", "s1", "prompt", None, (), 4000, frozenset(), False))
    assert calls == ["postgres runbook: run pytest first"]
    assert results[0].component_ranks["structured"] == 1


def test_scrubber_placeholders_hold_no_structured_term():
    """The structured prefilter relies on this: scrubbing never adds a command or an issue key."""
    from recall_query import _COMMANDS
    from recall_scrub import scrub_text

    secrets = ("-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\nBearer tok123 "
               "postgres://u:p@h/db API_KEY=v1 GITHUB_TOKEN=v2 password: v3 token=v4 /home/golden/x")
    scrubbed = scrub_text(secrets, Path("/home/golden")).casefold()
    assert scrubbed.count("<redacted>") >= 6 and "<home>" in scrubbed
    placeholders = " ".join(("<redacted>", "bearer <redacted>", "= <redacted>", "<home>"))
    assert not [term for term in _COMMANDS if term in placeholders]
    for term in _COMMANDS:
        assert term not in scrubbed or term in secrets.casefold()
    rows = synthetic_rows(300)
    for row in rows:
        folded, cleaned = row["text"].casefold(), scrub_text(row["text"], Path(GOLDEN_HOME)).casefold()
        for term in (*_COMMANDS, *extract_structured_terms(row["text"])):
            assert term not in cleaned or term in folded


def test_a_row_deleted_between_queries_is_dropped_not_looked_up(tmp_path, monkeypatch):
    """A writer's connection can lose a row after the FTS query; ranking skips it instead of failing."""
    import recall_query

    conn = _db(tmp_path)
    real_load_pool = recall_query._load_pool

    def load_pool_after_a_delete(conn, ids, *, stored):
        documents, path_texts = real_load_pool(conn, ids, stored=stored)
        return [doc for doc in documents if doc.id != "b"], path_texts

    monkeypatch.setattr(recall_query, "_load_pool", load_pool_after_a_delete)
    for query, dense in ((RecallQuery("postgres", "s1", "prompt", None, (), 4000, frozenset(), False), None),
                         (RecallQuery("pottery", "s1", "prompt", None, (), 4000, frozenset(), False), [("b", 0.9), ("a", 0.5)]),
                         (RecallQuery("/home/a/api", "s1", "session-start:startup", "/home/a/api", (), 6000, frozenset(), False), None)):
        results = rank_lexical(conn, query, dense=dense)
        assert [candidate.document.id for candidate in results] == ["a"]
