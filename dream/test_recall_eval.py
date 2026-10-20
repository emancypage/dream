import json
from pathlib import Path

import pytest

from recall_eval import calibrate_fixture_file, evaluate_fixture_file
from recall_types import AdapterSettings, RecallSettings


def _settings():
    return RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False))


def test_public_fixture_evaluates(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    report = evaluate_fixture_file(Path(__file__).parent / "fixtures/recall/public.json", conn, _settings())
    assert report.query_count == 3
    assert report.forbidden_count == 0
    assert report.recall_at_1 >= 0


def test_public_fixture_is_rejected_for_calibration(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    with pytest.raises(ValueError):
        calibrate_fixture_file(Path(__file__).parent / "fixtures/recall/public.json", conn, "lexical")


def test_private_fixture_calibration_has_stable_version(tmp_path):
    fixture = tmp_path / "heldout.json"
    fixture.write_text(json.dumps({"documents": [{"id":"a","source_kind":"approved_memory","trust_level":"user_approved","project_slug":None,"source_path":"a.md","source_updated_at":"2026-01-01T00:00:00Z","source_version":"v1","text":"alpha"}], "queries": [{"id":"q","event":"prompt","query":"alpha","relevant":["a"]}]}), encoding="utf-8")
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    record = calibrate_fixture_file(fixture, conn, "lexical")
    assert record.mode == "lexical"
    assert len(record.calibration_version) == 64


def test_recall_eval_cli_uses_ephemeral_database(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_eval

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_path = db_dir / "dream.db"
    db_dir.chmod(0o555)
    try:
        rc = cmd_recall_eval(argparse.Namespace(
            fixtures=Path(__file__).parent / "fixtures/recall/public.json",
            db=db_path,
            config=load_config(tmp_path / "missing.toml"),
        ))
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["query_count"] == 3
    finally:
        db_dir.chmod(0o755)


class KeywordEmbedder:
    fingerprint = "keyword-v1"
    remote_data_egress = False

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "postgres" in text.casefold() else [0.0, 1.0] for text in texts]


class BrokenEmbedder:
    fingerprint = "broken-v1"
    remote_data_egress = False

    def embed(self, texts):
        raise RuntimeError("offline")


def _language_fixture(tmp_path):
    fixture = tmp_path / "heldout.json"
    fixture.write_text(json.dumps({
        "documents": [
            {"id": "doc-api", "source_kind": "approved_memory", "trust_level": "user_approved", "project_slug": None,
             "source_path": "api.md", "source_updated_at": "2026-01-01T00:00:00Z", "source_version": "v1",
             "text": "The API project uses PostgreSQL migrations and pytest."},
            {"id": "doc-web", "source_kind": "approved_memory", "trust_level": "user_approved", "project_slug": None,
             "source_path": "web.md", "source_updated_at": "2026-01-01T00:00:00Z", "source_version": "v1",
             "text": "The web project uses npm and a deployment checklist."},
        ],
        "queries": [
            {"id": "q-en-1", "event": "prompt", "query": "postgresql migrations", "language": "en", "relevant": ["doc-api"]},
            {"id": "q-en-2", "event": "prompt", "query": "npm deployment checklist", "language": "en", "relevant": ["doc-web"]},
            {"id": "q-pl-1", "event": "prompt", "query": "jaka baza danych", "language": "pl", "relevant": ["doc-api"]},
        ],
    }), encoding="utf-8")
    return fixture


_REPORT_FIELDS = {
    "recall_at_1", "recall_at_3", "recall_at_5", "mrr", "injected_result_precision", "forbidden_count",
    "empty_result_correctness", "duplicate_count", "codepoints", "p50_latency_ms", "p95_latency_ms",
    "calibration_version", "query_count",
}
_SLICE_FIELDS = {"recall_at_1", "recall_at_5", "mrr", "forbidden_count", "p95_latency_ms", "query_count", "fallback_count",
                 "retrieval_recall_at_1", "retrieval_recall_at_5", "retrieval_mrr", "retrieval_forbidden_count"}


def test_report_breaks_down_by_language_and_mode(tmp_path):
    from dream import open_db
    embedder = KeywordEmbedder()
    report = evaluate_fixture_file(_language_fixture(tmp_path), open_db(tmp_path / "dream.db"), _settings(), embedder=embedder)
    breakdown = report.breakdown_dict()
    assert set(breakdown) == {"lexical", "lexical_plus_embedder"}
    assert embedder.calls > 0
    for mode in breakdown.values():
        assert set(mode) == {"all", "en", "pl"}
        assert all(set(values) == _SLICE_FIELDS for values in mode.values())
    lexical = breakdown["lexical"]
    assert lexical["en"]["query_count"] == 2 and lexical["pl"]["query_count"] == 1
    assert lexical["en"]["recall_at_1"] == 1.0 and lexical["en"]["recall_at_5"] == 1.0 and lexical["en"]["mrr"] == 1.0
    assert lexical["pl"]["recall_at_5"] == 0.0
    assert lexical["all"]["query_count"] == 3
    assert breakdown["lexical_plus_embedder"]["all"]["fallback_count"] == 0
    # Top-level fields are unchanged and still describe the lexical mode.
    assert set(report.to_dict()) == _REPORT_FIELDS
    assert report.recall_at_5 == pytest.approx(2 / 3)


def test_breakdown_without_embedder_has_only_lexical_mode(tmp_path):
    from dream import open_db
    report = evaluate_fixture_file(_language_fixture(tmp_path), open_db(tmp_path / "dream.db"), _settings())
    assert set(report.breakdown_dict()) == {"lexical"}


def test_embedder_failure_is_counted_not_hidden(tmp_path):
    from dream import open_db
    report = evaluate_fixture_file(_language_fixture(tmp_path), open_db(tmp_path / "dream.db"), _settings(), embedder=BrokenEmbedder())
    assert report.breakdown_dict()["lexical_plus_embedder"]["all"]["fallback_count"] == 3


def test_reserved_aggregate_language_is_rejected(tmp_path):
    from dream import open_db
    fixture = _language_fixture(tmp_path)
    data = json.loads(fixture.read_text(encoding="utf-8"))
    data["queries"][0]["language"] = "all"
    fixture.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="reserved"):
        evaluate_fixture_file(fixture, open_db(tmp_path / "dream.db"), _settings())


def test_untagged_public_queries_are_grouped(tmp_path):
    from dream import open_db
    report = evaluate_fixture_file(Path(__file__).parent / "fixtures/recall/public.json", open_db(tmp_path / "dream.db"), _settings())
    assert set(report.breakdown_dict()["lexical"]) == {"all", "untagged"}
    assert report.breakdown_dict()["lexical"]["all"]["forbidden_count"] == 0


def test_expected_ids_is_read_as_an_alias_for_relevant(tmp_path):
    from dream import open_db
    fixture = _language_fixture(tmp_path)
    data = json.loads(fixture.read_text(encoding="utf-8"))
    for query in data["queries"]:
        query["expected_ids"] = query.pop("relevant")
    alias = tmp_path / "alias.json"
    alias.write_text(json.dumps(data), encoding="utf-8")
    left = evaluate_fixture_file(fixture, open_db(tmp_path / "a.db"), _settings())
    right = evaluate_fixture_file(alias, open_db(tmp_path / "b.db"), _settings())
    assert (left.recall_at_1, left.recall_at_5, left.mrr) == (right.recall_at_1, right.recall_at_5, right.mrr)
    assert left.recall_at_5 > 0


def test_calibration_refuses_any_fixture_under_repository_fixtures(tmp_path):
    from dream import open_db
    conn = open_db(tmp_path / "dream.db")
    fixtures = Path(__file__).resolve().parent / "fixtures"
    for candidate in (fixtures / "recall" / "heldout.json", fixtures / "private-eval.json"):
        with pytest.raises(ValueError, match="public evaluation fixtures cannot be used for calibration"):
            calibrate_fixture_file(candidate, conn, "lexical")
        assert not candidate.exists()
    link = tmp_path / "heldout-link.json"
    link.symlink_to(fixtures / "recall" / "public.json")
    with pytest.raises(ValueError, match="public evaluation fixtures cannot be used for calibration"):
        calibrate_fixture_file(link, conn, "lexical")


def test_recall_calibrate_cli_never_opens_the_live_database(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_calibrate

    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_dir.chmod(0o555)
    try:
        rc = cmd_recall_calibrate(argparse.Namespace(
            fixtures=_language_fixture(tmp_path), db=db_dir / "dream.db", mode="lexical", output=None,
            config=load_config(tmp_path / "missing.toml"),
        ))
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["mode"] == "lexical"
        assert list(db_dir.iterdir()) == []
    finally:
        db_dir.chmod(0o755)


def test_recall_eval_cli_by_language_adds_breakdown_only_on_request(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_eval

    fixture = _language_fixture(tmp_path)
    config = load_config(tmp_path / "missing.toml")
    assert cmd_recall_eval(argparse.Namespace(fixtures=fixture, db=tmp_path / "dream.db", config=config)) == 0
    plain = json.loads(capsys.readouterr().out)
    assert set(plain) == _REPORT_FIELDS
    assert cmd_recall_eval(argparse.Namespace(fixtures=fixture, db=tmp_path / "dream.db", config=config, by_language=True)) == 0
    detailed = json.loads(capsys.readouterr().out)
    assert {key: detailed[key] for key in _REPORT_FIELDS if "latency" not in key} == {key: plain[key] for key in _REPORT_FIELDS if "latency" not in key}
    assert set(detailed["breakdown"]["lexical"]) == {"all", "en", "pl"}
    assert detailed["breakdown"]["lexical"]["pl"]["recall_at_5"] == 0.0


def test_recall_eval_parser_accepts_by_language(tmp_path, capsys, monkeypatch):
    import dream

    monkeypatch.setenv("HOME", str(tmp_path))
    rc = dream.main(["--config", str(tmp_path / "missing.toml"), "recall-eval", "--fixtures", str(_language_fixture(tmp_path)), "--by-language"])
    assert rc == 0
    assert "breakdown" in json.loads(capsys.readouterr().out)


def test_embedder_mode_embeds_fixture_documents_into_the_eval_database(tmp_path):
    from dream import open_db
    embedder = KeywordEmbedder()
    conn = open_db(tmp_path / "dream.db")
    evaluate_fixture_file(_language_fixture(tmp_path), conn, _settings(), embedder=embedder)
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings WHERE adapter_fingerprint='keyword-v1'").fetchone()[0] == 2


def test_recall_eval_cli_reports_the_configured_embedder_mode(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_eval
    from test_recall_embedder_http import DIMENSION, MODEL, FakeEmbeddingServer

    server = FakeEmbeddingServer().start()
    db_dir = tmp_path / "store"
    db_dir.mkdir()
    db_dir.chmod(0o555)
    try:
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            "[recall.embedder]\nenabled = true\ntype = \"openai-compatible\"\n"
            f"base_url = \"{server.url}\"\nmodel = \"{MODEL}\"\ndimension = {DIMENSION}\n", encoding="utf-8")
        rc = cmd_recall_eval(argparse.Namespace(
            fixtures=_language_fixture(tmp_path), db=db_dir / "dream.db", config=load_config(config_path), by_language=True,
        ))
        assert rc == 0
        breakdown = json.loads(capsys.readouterr().out)["breakdown"]
        assert set(breakdown) == {"lexical", "lexical_plus_embedder"}
        assert breakdown["lexical"]["pl"]["recall_at_5"] == 0.0
        assert breakdown["lexical_plus_embedder"]["pl"]["recall_at_5"] == 1.0
        assert breakdown["lexical_plus_embedder"]["en"]["recall_at_5"] == 1.0
        assert breakdown["lexical_plus_embedder"]["all"]["fallback_count"] == 0
        posted = server.posted_inputs()
        assert any("PostgreSQL migrations" in text for text in posted)
        assert "jaka baza danych" in posted
        assert list(db_dir.iterdir()) == []
    finally:
        db_dir.chmod(0o755)
        server.stop()


def test_recall_eval_cli_reports_lexical_only_when_the_embedder_is_misconfigured(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_eval

    config_path = tmp_path / "config.toml"
    config_path.write_text('[recall.embedder]\nenabled = true\ntype = "openai-compatible"\nbase_url = "http://192.0.2.10:8081"\n', encoding="utf-8")
    rc = cmd_recall_eval(argparse.Namespace(fixtures=_language_fixture(tmp_path), db=tmp_path / "dream.db",
                                            config=load_config(config_path), by_language=True))
    captured = capsys.readouterr()
    assert rc == 0
    assert set(json.loads(captured.out)["breakdown"]) == {"lexical"}
    assert "remote_data_egress" in captured.err


def test_embedder_mode_calibration_needs_an_embedder_and_sweeps_fused_scores(tmp_path, monkeypatch):
    from dream import open_db
    fixture = _language_fixture(tmp_path)
    with pytest.raises(ValueError, match="embedder"):
        calibrate_fixture_file(fixture, open_db(tmp_path / "a.db"), "lexical_plus_embedder")
    lexical = calibrate_fixture_file(fixture, open_db(tmp_path / "b.db"), "lexical")
    embedder = KeywordEmbedder()
    import recall_eval
    swept = []
    real_key = recall_eval.calibration_key
    monkeypatch.setattr(recall_eval, "calibration_key", lambda rows, threshold: swept.append(threshold) or real_key(rows, threshold))
    record = calibrate_fixture_file(fixture, open_db(tmp_path / "c.db"), "lexical_plus_embedder", embedder=embedder)
    assert record.mode == "lexical_plus_embedder"
    assert record.calibration_version != lexical.calibration_version
    assert record.fixture_sha256 == lexical.fixture_sha256
    assert embedder.calls > 0
    # The sweep covers exactly the fused, rescored scores of the embedder mode (no artificial 0.0).
    from dream import _open_ephemeral_eval_db
    from recall_context import retrieve
    conn = _open_ephemeral_eval_db()
    data = json.loads(fixture.read_text(encoding="utf-8"))
    recall_eval._ensure_documents(conn, data["documents"])
    recall_eval._embed_fixture_documents(conn, KeywordEmbedder(), _settings())
    expected = set()
    for item in data["queries"]:
        outcome = retrieve(conn, recall_eval._query(item, _settings()), recall_eval._embedder_settings(_settings()), {},
                           embedder=KeywordEmbedder(), require_calibration=False)
        assert outcome.mode == "lexical_plus_embedder"
        expected.update(candidate.score for candidate in outcome.candidates)
    assert swept == sorted(expected) and 0.0 not in swept
    assert 0 < record.threshold <= max(swept)

    class OtherModel(KeywordEmbedder):
        fingerprint = "keyword-v2"

    other = calibrate_fixture_file(fixture, open_db(tmp_path / "d.db"), "lexical_plus_embedder", embedder=OtherModel())
    assert other.calibration_version != record.calibration_version
    assert other.fixture_sha256 == record.fixture_sha256


def test_recall_calibrate_cli_refuses_embedder_mode_without_an_embedder(tmp_path, capsys):
    import argparse

    from config import load_config
    from dream import cmd_recall_calibrate

    rc = cmd_recall_calibrate(argparse.Namespace(fixtures=_language_fixture(tmp_path), db=tmp_path / "dream.db",
                                                 config=load_config(tmp_path / "missing.toml"), mode="lexical_plus_embedder", output=None))
    assert rc == 2
    assert "embedder" in capsys.readouterr().err
    assert not (tmp_path / "dream.db").exists()


# --- calibration that is precise and fast (eval-build shaped fixtures) ----------

_TOPICS = [("postgres database migration", "baza danych migracja"), ("npm deploy release", "wdrożenie wydanie"),
           ("docker container image", "kontener obraz"), ("pytest unit tests", "testy jednostkowe"),
           ("redis cache eviction", "pamięć podręczna"), ("kubernetes cluster nodes", "klaster węzły")]


class TopicEmbedder:
    """Real models give unrelated texts a positive cosine: every vector shares a common component."""

    fingerprint = "topic:16"
    remote_data_egress = False

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        import hashlib
        import re
        self.calls += 1
        vectors = []
        for text in texts:
            words = set(re.findall(r"\w+", text.casefold()))
            vector = [0.3] * 16
            for axis, (english, polish) in enumerate(_TOPICS):
                if words & (set(english.split()) | set(polish.split())):
                    vector[axis] += 1.0
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([value + digest[index] / 255 * 0.1 for index, value in enumerate(vector)])
        return vectors


def _topic_fixture(tmp_path, queries=30, documents=40):
    import random
    rng = random.Random(3)
    docs, items = [], []
    for index in range(documents):
        english, _polish = _TOPICS[index % len(_TOPICS)]
        words = " ".join(rng.sample(["alpha", "beta", "gamma", "delta", "omega"], 2))
        docs.append({"id": f"doc-{index}", "source_kind": "approved_memory", "trust_level": "user_approved", "project_slug": None,
                     "source_path": f"t{index}.md", "source_updated_at": f"2026-01-{1 + index % 28:02d}T00:00:00Z",
                     "source_version": "v1", "text": f"Note {index}: we use {english} with extra words {words}."})
    for number in range(queries):
        index = rng.randrange(documents)
        english, polish = _TOPICS[index % len(_TOPICS)]
        language = "pl" if number % 2 else "en"
        items.append({"id": f"q{number}", "event": "prompt", "query": (polish if language == "pl" else english) + f" pytanie {number}",
                      "language": language, "relevant": [f"doc-{index}"], "forbidden": []})
    fixture = tmp_path / f"heldout-{queries}x{documents}.json"
    fixture.write_text(json.dumps({"documents": docs, "queries": items}), encoding="utf-8")
    return fixture, docs


def test_embedder_calibration_is_precise_and_fast_on_an_eval_build_shaped_fixture(tmp_path):
    import time
    from dream import _open_ephemeral_eval_db
    from recall_context import retrieve
    from recall_select import select_recall_candidates
    import recall_eval
    fixture, docs = _topic_fixture(tmp_path)
    embedder = TopicEmbedder()
    started = time.perf_counter()
    record = calibrate_fixture_file(fixture, _open_ephemeral_eval_db(), "lexical_plus_embedder", embedder=embedder, require_usable=True)
    assert time.perf_counter() - started < 2.0
    assert record.threshold > 0
    # An unrelated prompt now selects far fewer than the 40 documents a 0.0 threshold let through.
    conn = _open_ephemeral_eval_db()
    recall_eval._ensure_documents(conn, docs)
    recall_eval._embed_fixture_documents(conn, embedder, _settings())
    calibration = {"lexical_plus_embedder": {"threshold": record.threshold}}
    for text in ("pottery glazing in the studio", "garncarstwo i szkliwienie"):
        query = recall_eval._query({"query": text, "event": "prompt"}, _settings())
        outcome = retrieve(conn, query, recall_eval._embedder_settings(_settings()), calibration, embedder=embedder,
                           require_calibration=False)
        assert outcome.mode == "lexical_plus_embedder"
        assert len(outcome.candidates) == 40
        assert len(outcome.selected) < 5
        assert len(select_recall_candidates(outcome.candidates, query, None, {"lexical_plus_embedder": {"threshold": 0.0}},
                                            mode="lexical_plus_embedder")) == 40


def test_cached_calibration_sweep_is_as_good_as_the_naive_one(tmp_path):
    import bisect
    from dream import _open_ephemeral_eval_db
    import recall_eval
    fixture, _docs = _topic_fixture(tmp_path, queries=6, documents=8)
    for mode, embedder in (("lexical", None), ("lexical_plus_embedder", TopicEmbedder())):
        record = calibrate_fixture_file(fixture, _open_ephemeral_eval_db(), mode, embedder=embedder)
        # Naive reference: the full pipeline, every query, at every swept threshold.
        conn = _open_ephemeral_eval_db()
        data = json.loads(fixture.read_text(encoding="utf-8"))
        recall_eval._ensure_documents(conn, data["documents"])
        settings = recall_eval._lexical_settings(_settings())
        if embedder is not None:
            recall_eval._embed_fixture_documents(conn, embedder, _settings())
            settings = recall_eval._embedder_settings(_settings())
        swept = set() if embedder is not None else {0.0}
        for item in data["queries"]:
            outcome = recall_eval.retrieve(conn, recall_eval._query(item, _settings()), settings, {}, embedder=embedder,
                                           require_calibration=False)
            swept.update(candidate.score for candidate in outcome.candidates)

        def naive_key(threshold):
            rows = recall_eval._run_queries(data, conn, settings, {mode: {"threshold": threshold}}, embedder=embedder)
            cuts = []
            for item, row in zip(data["queries"], rows):
                ids = row["delivered_ids"]
                relevant = row["relevant"]
                cuts.append((bool(relevant), recall_eval._Cut(
                    min(row["ranks"]) if row["ranks"] else None,
                    len(relevant.intersection(ids)) / len(ids) if ids else 0.0, row["forbidden"])))
            return recall_eval.calibration_key(cuts, threshold)[:3]

        best = max(naive_key(threshold) for threshold in sorted(swept))
        assert naive_key(record.threshold) == best, mode
        ordered = sorted(swept)
        chosen = ordered[bisect.bisect_left(ordered, record.threshold)] if record.threshold > 0 else 0.0
        assert naive_key(chosen) == best


def test_apply_refuses_a_zero_threshold_for_the_embedder_mode(tmp_path, monkeypatch):
    from dataclasses import replace as dc_replace
    from dream import _open_ephemeral_eval_db
    import recall_eval
    fixture, _docs = _topic_fixture(tmp_path, queries=4, documents=6)
    real_retrieve = recall_eval.retrieve

    def zero_scores(*args, **kwargs):
        outcome = real_retrieve(*args, **kwargs)
        return dc_replace(outcome, candidates=tuple(dc_replace(candidate, score=0.0) for candidate in outcome.candidates))

    monkeypatch.setattr(recall_eval, "retrieve", zero_scores)
    with pytest.raises(ValueError, match="threshold of 0"):
        calibrate_fixture_file(fixture, _open_ephemeral_eval_db(), "lexical_plus_embedder", embedder=TopicEmbedder(), require_usable=True)
    # Lexical keeps today's behaviour: a 0.0 threshold is accepted.
    assert calibrate_fixture_file(fixture, _open_ephemeral_eval_db(), "lexical", require_usable=True).threshold == 0.0


# --- gate metrics score the context an agent actually receives ------------------


def _budget_fixture(tmp_path):
    """Five long, equally ranked matches; the relevant one is fifth and the first four fill 4,000 code points."""
    documents = [{"id": str(index), "source_kind": "approved_memory", "trust_level": "user_approved", "project_slug": None,
                  "source_path": f"{index}.md", "source_updated_at": "2026-01-01T00:00:00Z", "source_version": "v1",
                  "text": f"alpha record {index} " + "details " * 200} for index in range(5)]
    queries = [
        {"id": "relevant-fifth", "event": "prompt", "query": "alpha", "language": "en", "relevant": ["4"]},
        {"id": "forbidden-fifth", "event": "prompt", "query": "alpha", "language": "en", "relevant": ["0"], "forbidden": ["4"]},
    ]
    fixture = tmp_path / "budget.json"
    fixture.write_text(json.dumps({"documents": documents, "queries": queries}), encoding="utf-8")
    return fixture, {"documents": documents, "queries": queries}


def test_gate_metrics_count_only_documents_inside_the_rendered_budget(tmp_path):
    from dream import _open_ephemeral_eval_db
    from recall_context import recall_for_agent
    from recall_eval import _ensure_documents, _run_queries
    fixture, data = _budget_fixture(tmp_path)
    conn = _open_ephemeral_eval_db()
    _ensure_documents(conn, data["documents"])
    first, second = _run_queries(data, conn, _settings(), {})
    assert first["selected_ids"] == ["0", "1", "2", "3", "4"]
    assert first["delivered_ids"] == ["0", "1", "2", "3"]
    assert first["ranks"] == [] and first["retrieval_ranks"] == [5]
    assert second["forbidden"] == 0 and second["retrieval_forbidden"] == 1
    # The evaluator delivers exactly what the agent query renders.
    agent = recall_for_agent(conn, _settings(), query_text="alpha", cwd=None, memory_root=None, constitution_path=None, limit=10)
    assert [candidate.document.id for candidate in agent.selected] == first["delivered_ids"]
    assert first["codepoints"] == len(agent.rendered_context) == 4000

    report = evaluate_fixture_file(fixture, _open_ephemeral_eval_db(), _settings())
    lexical = report.breakdown_dict()["lexical"]["all"]
    assert (report.recall_at_5, report.mrr, report.forbidden_count) == (0.5, 0.5, 0)
    assert (lexical["recall_at_5"], lexical["retrieval_recall_at_5"]) == (0.5, 1.0)
    assert (lexical["forbidden_count"], lexical["retrieval_forbidden_count"]) == (0, 1)
    assert report.retrieval == {"recall_at_1": 0.5, "recall_at_3": 0.5, "recall_at_5": 1.0, "mrr": pytest.approx(0.6),
                                "forbidden_count": 1}
    assert set(report.to_dict()) == _REPORT_FIELDS


def test_recall_eval_cli_keeps_its_default_shape_and_adds_retrieval_numbers_with_by_language(tmp_path, capsys):
    import argparse
    from config import load_config
    from dream import cmd_recall_eval
    fixture, _data = _budget_fixture(tmp_path)
    config = load_config(tmp_path / "missing.toml")
    assert cmd_recall_eval(argparse.Namespace(fixtures=fixture, db=tmp_path / "dream.db", config=config)) == 0
    plain = json.loads(capsys.readouterr().out)
    assert set(plain) == _REPORT_FIELDS and plain["recall_at_5"] == 0.5
    assert cmd_recall_eval(argparse.Namespace(fixtures=fixture, db=tmp_path / "dream.db", config=config, by_language=True)) == 0
    detailed = json.loads(capsys.readouterr().out)
    assert detailed["retrieval"]["recall_at_5"] == 1.0
    assert detailed["breakdown"]["lexical"]["en"]["retrieval_recall_at_5"] == 1.0


def _stub_candidate(doc_id, score, *, long):
    from recall_types import RecallCandidate, RecallDocument
    text = f"note {doc_id} " + ("filler words here " * 80 if long else "short")
    document = RecallDocument(doc_id, f"sha-{doc_id}", "approved_memory", "user_approved", None, f"{doc_id}.md",
                              "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "v1", text)
    return RecallCandidate(document, text, {"fts": 1}, score, 1)


def test_calibration_scores_delivered_context_and_that_changes_the_threshold():
    from types import SimpleNamespace
    from recall_eval import _query, _query_cuts, _sweep
    settings = _settings()
    # Q1: four long irrelevant documents outscore the relevant one, which is selected fifth but never rendered.
    q1 = {"query": "alpha", "relevant": ["r1"]}
    q1_outcome = SimpleNamespace(mode="lexical", candidates=tuple(
        [_stub_candidate(f"a{index}", 0.5, long=True) for index in range(4)] + [_stub_candidate("r1", 0.3, long=False)]))
    # Q2: the relevant document leads; a short irrelevant one follows at a lower score.
    q2 = {"query": "beta", "relevant": ["r2"]}
    q2_outcome = SimpleNamespace(mode="lexical", candidates=(_stub_candidate("r2", 0.6, long=False), _stub_candidate("n2", 0.35, long=False)))
    per_query = []
    for item, outcome in ((q1, q1_outcome), (q2, q2_outcome)):
        cuts, rows = _query_cuts(item, _query(item, settings), outcome, settings, "lexical")
        per_query.append((True, cuts, rows))
    # Q1's relevant document is never delivered, at any threshold.
    assert all(row.best_rank is None for row in per_query[0][2])
    key, threshold = _sweep(per_query, embedder_mode=False)
    # Scored on selection, 0.3 would win (Q1 at rank 5 adds MRR 0.2). Delivered, Q1 never counts, so the
    # threshold that drops Q2's irrelevant document wins: the middle of (0.35, 0.5].
    assert threshold == pytest.approx(0.425)
    assert key[:3] == (0.5, 0.5, 0.5)
