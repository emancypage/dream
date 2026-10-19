import argparse
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import dream
import eval_build
from config import load_config
from model_types import ProviderError
from recall_documents import stable_document_id, synchronize_recall_documents
from recall_eval import calibrate_fixture_file, evaluate_fixture_file
from retention_policy import POLICY_FILE


POLICY_INSTRUCTIONS = "Do not retain the retired project, including paraphrases."


def _answers(stage, prompt, schema, **kwargs):
    return {
        "en": [
            "Which database does that project use?",
            "which database does that project use?",
            "How do I reach the retired-marker host?",
            "Where is password=hunter2 stored?",
            "What runs the tests?",
            "How are releases shipped?",
            "One question too many?",
        ],
        "pl": ["Jakiej bazy danych używa projekt?", "Jak uruchamiam testy?", "Jak wdrażamy wydania?"],
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "memory"
    root.mkdir()
    (root / "api.md").write_text("The API project uses PostgreSQL migrations and pytest.", encoding="utf-8")
    (root / "web.md").write_text("The web project ships releases through a deployment checklist.", encoding="utf-8")
    (root / "private.md").write_text("Excluded by path: the old staging host lives in room 12.", encoding="utf-8")
    (root / "marked.md").write_text("Notes about the retired-marker deployment.", encoding="utf-8")
    (root / "MEMORY.md").write_text("- [API](api.md) — database\n- [Web](web.md) — releases\n", encoding="utf-8")
    conn = dream.open_db(tmp_path / "dream.db")
    conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime,total_chars) "
                 "VALUES ('ordinary-session','ordinary-project','/fixture/a',1,1000)")
    conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime,total_chars) "
                 "VALUES ('private-session','private-project','/fixture/b',1,1000)")
    conn.execute("INSERT INTO distilled(session_id,notes_json,summary,distillation_key) VALUES (?,?,?,?)",
                 ("ordinary-session", json.dumps({"summary": "Moved the web project to blue-green releases"}),
                  "Moved the web project to blue-green releases", "ordinary-key"))
    conn.execute("INSERT INTO distilled(session_id,notes_json,summary,distillation_key) VALUES (?,?,?,?)",
                 ("private-session", json.dumps({"summary": "Excluded by project: the vault rotation plan"}),
                  "Excluded by project: the vault rotation plan", "private-key"))
    conn.commit()
    # Index first, then tighten the policy: the builder must re-apply it to stale rows.
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, constitution_path=None)
    (root / POLICY_FILE).write_text(json.dumps({
        "instructions": POLICY_INSTRUCTIONS,
        "memory_paths": ["private.md"],
        "project_slugs": ["private-project"],
        "text_markers": ["retired-marker"],
    }), encoding="utf-8")
    cfg = load_config(tmp_path / "absent.toml", {"storage": {"memory_root": str(root), "db_path": str(tmp_path / "dream.db")}})
    calls = []

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append(SimpleNamespace(stage=stage, prompt=prompt, schema=schema, kwargs=kwargs))
        return SimpleNamespace(output=_answers(stage, prompt, schema), provider="fake", model="fake-model")

    monkeypatch.setattr(eval_build, "generate", fake_generate)
    yield SimpleNamespace(conn=conn, root=root, cfg=cfg, calls=calls, tmp=tmp_path, home=home)
    conn.close()


def test_fixture_has_public_shape_and_language_tags(store):
    output = store.tmp / "out" / "private.json"
    fixture = eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=output)

    assert json.loads(output.read_text(encoding="utf-8")) == fixture
    assert isinstance(fixture["documents"], list) and isinstance(fixture["queries"], list)
    ids = {document["id"] for document in fixture["documents"]}
    api_id = stable_document_id("approved_memory", "memory:api.md")
    summary_id = stable_document_id("distilled_summary", "distilled:ordinary-key")
    assert ids == {api_id, stable_document_id("approved_memory", "memory:web.md"),
                   stable_document_id("approved_memory", "memory:MEMORY.md"), summary_id}
    for document in fixture["documents"]:
        assert {"id", "content_sha256", "source_kind", "trust_level", "project_slug", "source_path",
                "source_updated_at", "indexed_at", "source_version", "text"} == set(document)
    summary = next(document for document in fixture["documents"] if document["id"] == summary_id)
    assert summary["project_slug"] == "ordinary-project"
    assert summary["trust_level"] == "model_distilled"

    # Three documents receive questions; MEMORY.md is corpus only.
    assert len(store.calls) == 3
    assert len(fixture["queries"]) == 3 * 2 * 3
    assert {query["language"] for query in fixture["queries"]} == {"en", "pl"}
    assert len({query["id"] for query in fixture["queries"]}) == len(fixture["queries"])
    for query in fixture["queries"]:
        assert query["event"] == "prompt"
        assert "expected_ids" not in query
        assert query["relevant"] and set(query["relevant"]) <= ids
        assert stable_document_id("approved_memory", "memory:MEMORY.md") not in query["relevant"]
    api_questions = [query["query"] for query in fixture["queries"]
                     if query["relevant"] == [api_id] and query["language"] == "en"]
    # Duplicates, policy matches and secret-bearing questions are dropped, then capped.
    assert api_questions == ["Which database does that project use?", "What runs the tests?", "How are releases shipped?"]
    assert fixture["generator"]["languages"] == ["en", "pl"]
    assert fixture["generator"]["failed_documents"] == 0

    # The fixture round-trips through the evaluator.
    report = evaluate_fixture_file(output, dream.open_db(store.tmp / "eval.db"))
    assert report.query_count == 18


def test_provider_is_the_distill_stage_with_a_question_schema(store):
    eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=store.tmp / "private.json",
                                  languages=("en", "pl"), questions_per_file=2)
    assert {call.stage for call in store.calls} == {"distill"}
    for call in store.calls:
        assert call.kwargs["config"] is store.cfg
        assert call.schema == {
            "type": "object",
            "required": ["en", "pl"],
            "additionalProperties": False,
            "properties": {
                "en": {"type": "array", "items": {"type": "string"}},
                "pl": {"type": "array", "items": {"type": "string"}},
            },
        }
        assert "2 distinct questions" in call.prompt
        assert "pl (Polish)" in call.prompt


def test_documents_are_copied_verbatim_so_calibration_upsert_is_a_no_op(store):
    output = store.tmp / "private.json"
    fixture = eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=output)
    columns = ("id, content_sha256, source_kind, trust_level, project_slug, source_path, "
               "source_updated_at, indexed_at, source_version, text")
    rows = {row[0]: row for row in store.conn.execute(f"SELECT {columns} FROM recall_documents")}
    for document in fixture["documents"]:
        assert tuple(document[key] for key in columns.split(", ")) == rows[document["id"]]
    before = store.conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()
    calibrate_fixture_file(output, store.conn, "lexical")
    assert store.conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall() == before


def test_excluded_documents_are_skipped_and_never_sent(store):
    fixture = eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=store.tmp / "private.json")
    sent = "\n".join(call.prompt for call in store.calls)
    for excluded in ("room 12", "vault rotation", "Notes about the retired-marker deployment"):
        assert excluded not in sent
        assert excluded not in json.dumps(fixture, ensure_ascii=False)
    paths = {document["source_path"] for document in fixture["documents"]}
    assert not paths & {"private.md", "marked.md", "distilled:private-key"}
    assert all(POLICY_INSTRUCTIONS in call.prompt for call in store.calls)
    assert not any("retired-marker" in query["query"] for query in fixture["queries"])


def test_documents_are_scrubbed_before_they_reach_the_provider(store):
    (store.root / "token.md").write_text(f"Deploy with API_TOKEN=abc123secret from {store.home}/deploy", encoding="utf-8")
    synchronize_recall_documents(store.conn, store.root, include_raw_transcripts=False, constitution_path=None)
    eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=store.tmp / "private.json")
    sent = "\n".join(call.prompt for call in store.calls)
    assert "abc123secret" not in sent
    assert str(store.home) not in sent


def test_limit_samples_both_document_kinds_and_keeps_full_corpus(store):
    fixture = eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=store.tmp / "private.json", limit=2)
    assert len(store.calls) == 2
    kinds = {document["id"]: document["source_kind"] for document in fixture["documents"]}
    asked = {query["relevant"][0] for query in fixture["queries"]}
    assert {kinds[doc_id] for doc_id in asked} == {"approved_memory", "distilled_summary"}
    assert len(fixture["documents"]) == 4
    assert fixture["generator"]["sampled_documents"] == 2


def test_provider_failure_skips_only_that_document(store, monkeypatch):
    def flaky(stage, prompt, schema, **kwargs):
        store.calls.append(prompt)
        if len(store.calls) == 1:
            raise ProviderError("offline")
        return SimpleNamespace(output=_answers(stage, prompt, schema), provider="fake", model="fake-model")

    monkeypatch.setattr(eval_build, "generate", flaky)
    fixture = eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=store.tmp / "private.json")
    assert fixture["generator"]["failed_documents"] == 1
    assert len({query["relevant"][0] for query in fixture["queries"]}) == 2


def test_fixture_is_written_privately_to_the_default_path(store):
    target = store.home / ".config" / "dream" / "eval" / "private.json"
    assert eval_build.default_output_path() == target
    target.parent.mkdir(parents=True)
    target.parent.chmod(0o755)
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o644)

    eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) & 0o077 == 0
    assert json.loads(target.read_text(encoding="utf-8"))["queries"]
    assert [path.name for path in target.parent.iterdir()] == ["private.json"]


def test_new_output_directories_are_private(store):
    output = store.tmp / "fresh" / "nested" / "private.json"
    eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    for directory in (output.parent, output.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) & 0o077 == 0


def test_invalid_languages_and_counts_are_rejected(store):
    for languages in ((), ("en", "en"), ("English",), ("all",), ("untagged",)):
        with pytest.raises(ValueError):
            eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg,
                                          output=store.tmp / "private.json", languages=languages)
    with pytest.raises(ValueError):
        eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg,
                                      output=store.tmp / "private.json", questions_per_file=0)
    assert not store.calls
    assert not (store.tmp / "private.json").exists()


def test_output_inside_the_repository_is_refused_before_any_call(store):
    target = Path(eval_build.__file__).resolve().parent / "fixtures" / "eval-build-test-output.json"
    with pytest.raises(ValueError, match="outside the repository"):
        eval_build.build_eval_fixture(store.conn, store.root, config=store.cfg, output=target)
    assert not store.calls
    assert not target.exists()


def test_gitignore_keeps_private_fixtures_out_of_the_repository():
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    assert "dream/fixtures/private*" in gitignore.read_text(encoding="utf-8").splitlines()


def _cli_args(store, **overrides):
    values = dict(db=store.tmp / "dream.db", config=store.cfg, limit=None, questions_per_file=3,
                  languages=("en", "pl"), output=str(store.tmp / "cli" / "private.json"), yes=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_eval_build_cli_requires_limit_or_yes(store, capsys):
    assert dream.cmd_eval_build(_cli_args(store)) == 2
    assert "3 documents" in capsys.readouterr().out
    assert not store.calls
    assert not (store.tmp / "cli" / "private.json").exists()


def test_eval_build_cli_writes_the_fixture(store, capsys):
    assert dream.cmd_eval_build(_cli_args(store, limit=2)) == 0
    output = store.tmp / "cli" / "private.json"
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["queries"]) == 2 * 2 * 3
    out = capsys.readouterr().out
    assert str(output) in out
    assert "PostgreSQL" not in out


def test_eval_build_cli_refuses_repository_output(store, capsys):
    target = Path(eval_build.__file__).resolve().parent / "eval-build-test-output.json"
    assert dream.cmd_eval_build(_cli_args(store, limit=1, output=str(target))) == 2
    assert "outside the repository" in capsys.readouterr().err
    assert not store.calls and not target.exists()


def test_eval_build_cli_refuses_an_empty_recall_index(store, capsys):
    store.conn.execute("DELETE FROM recall_documents")
    store.conn.commit()
    assert dream.cmd_eval_build(_cli_args(store, limit=5)) == 1
    assert "recall index" in capsys.readouterr().err
    assert not store.calls
    assert not (store.tmp / "cli" / "private.json").exists()


def test_eval_build_cli_reports_provider_failures(store, monkeypatch):
    monkeypatch.setattr(eval_build, "generate", lambda *a, **k: (_ for _ in ()).throw(ProviderError("offline")))
    assert dream.cmd_eval_build(_cli_args(store, yes=True)) == 1


def test_eval_build_parser_accepts_documented_flags(store, monkeypatch, capsys):
    config_path = store.tmp / "config.toml"
    config_path.write_text(f'[storage]\nmemory_root = "{store.root}"\ndb_path = "{store.tmp / "dream.db"}"\n', encoding="utf-8")
    output = store.tmp / "parsed.json"
    rc = dream.main(["--config", str(config_path), "eval-build", "--limit", "1", "--questions-per-file", "2",
                     "--languages", "pl,en", "--output", str(output)])
    assert rc == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["generator"]["languages"] == ["pl", "en"]
    assert len(data["queries"]) == 2 * 2
    with pytest.raises(SystemExit):
        dream.main(["--config", str(config_path), "eval-build", "--languages", "en,Polish!"])
