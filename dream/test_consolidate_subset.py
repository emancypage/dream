"""Subset consolidation: memory selection, the subset prompt, and the full-mode golden.

`consolidate.mode = "full"` stays byte-for-byte identical to the pre-subset pipeline.
The golden files under fixtures/consolidate/ were captured from the code before
subset consolidation, `patch` suggestions and the weekly audit existed. Set
DREAM_UPDATE_GOLDEN=1 only for an intentional, reviewed change to full mode.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import consolidate as C
import dream
from config import ConfigError, ConsolidateSettings, load_config
from recall_documents import synchronize_recall_documents


ROOT = Path(__file__).parent
SCHEMA = (ROOT / "schema.sql").read_text(encoding="utf-8")
GOLDEN_DIR = ROOT / "fixtures" / "consolidate"
GOLDEN_PROMPT = GOLDEN_DIR / "full-mode-prompt.txt"
GOLDEN_CALL = GOLDEN_DIR / "full-mode-call.json"
ROOT_PLACEHOLDER = "<memory-root>"


class _FixedDate(dt.date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 19)


def _fixed_clock(monkeypatch) -> None:
    monkeypatch.setattr(C, "dt", SimpleNamespace(**{**vars(dt), "date": _FixedDate}))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path, root, **consolidate):
    overrides = {"storage": {"db_path": str(tmp_path / "dream.db"), "memory_root": str(root)}}
    if consolidate:
        overrides["consolidate"] = consolidate
    return load_config(tmp_path / "missing-config.toml", overrides=overrides)


# ------------------------------------------------------------ full-mode golden


def _golden_memory_fixture(root: Path) -> None:
    root.mkdir()
    (root / "MEMORY.md").write_text(
        "- [Terse replies](feedback_terse_responses.md) — answer briefly\n"
        "- [Finance tracker](project_finances_tracker.md) — budget spreadsheet\n"
        "- [Grafana](reference_grafana.md) — dashboards\n",
        encoding="utf-8",
    )
    (root / "feedback_terse_responses.md").write_text(
        "---\nname: feedback-terse-responses\ndescription: answer briefly\nmetadata:\n  type: feedback\n---\n\n"
        "Keep answers short.\n**Why:** the user reads on a phone.\n**How to apply:** lead with the answer.\n",
        encoding="utf-8",
    )
    (root / "project_finances_tracker.md").write_text(
        "---\nname: project-finances-tracker\ndescription: budget spreadsheet\nmetadata:\n  type: project\n---\n\n"
        "The budget lives in a spreadsheet. ARCHIWALNE since the switch to the bank app.\n",
        encoding="utf-8",
    )
    (root / "reference_grafana.md").write_text(
        "---\nname: reference-grafana\ndescription: dashboards\nmetadata:\n  type: reference\n---\n\n"
        "Dashboards live on the build host.\n",
        encoding="utf-8",
    )
    (root / ".hidden.md").write_text("never shown", encoding="utf-8")


def _golden_db_fixture() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    for sid, ended, notes in (
        ("s-1", "2026-10-18T09:00:00Z", {
            "summary": "budget review", "user_facts": ["The user switched to a bank app for budgeting."],
            "preferences": ["Short answers."], "projects": ["project_finances_tracker.md is closed."],
            "references": [], "decisions": ["Keep Grafana dashboards."], "open_threads": [],
        }),
        ("s-2", "2026-10-18T10:00:00Z", {
            "summary": "grafana", "user_facts": [], "preferences": [],
            "projects": ["Grafana alert rules for the exporter."], "references": ["reference_grafana.md"],
            "decisions": [], "open_threads": ["tune alert thresholds"],
        }),
    ):
        conn.execute(
            "INSERT INTO sessions(session_id, source, external_session_id, project_slug, jsonl_path, "
            "jsonl_mtime, ended_at, total_chars) VALUES (?, 'codex', ?, '-home-user', '/p.jsonl', 1, ?, 1000)",
            (sid, sid, ended),
        )
        conn.execute(
            "INSERT INTO distilled(session_id, distillation_key, notes_json, summary) VALUES (?, ?, ?, ?)",
            (sid, f"key-{sid}", json.dumps(notes), notes["summary"]),
        )
    conn.commit()
    return conn


def _golden_model_output() -> dict:
    return {"suggestions": [
        {"kind": "update", "target_path": "reference_grafana.md",
         "body": "---\nname: reference-grafana\n---\n\nDashboards and alert rules live on the build host.\n",
         "rationale": "alert rules were added", "source_sessions": ["s-2"]},
        {"kind": "new", "target_path": "feedback_terse_responses.md",
         "body": "duplicate topic", "rationale": "model missed the existing file", "source_sessions": ["s-1"]},
        {"kind": "remove", "target_path": "project_finances_tracker.md", "body": "",
         "rationale": "ARCHIWALNE", "source_sessions": []},
        {"kind": "index", "target_path": "MEMORY.md",
         "body": "- [Alerts](reference_alerts.md) — alert rules\n", "rationale": "new file", "source_sessions": ["s-2"]},
    ]}


def _run_full_mode(tmp_path, monkeypatch, **consolidate_kwargs):
    _fixed_clock(monkeypatch)
    root = tmp_path / "memory"
    _golden_memory_fixture(root)
    conn = _golden_db_fixture()
    config = _config(tmp_path, root)
    calls = []

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append({"stage": stage, "prompt": prompt, "schema": schema, "kwargs": kwargs})
        return SimpleNamespace(output=_golden_model_output(), provider="fake", model="fake-model")

    monkeypatch.setattr(C, "generate", fake_generate)
    suggestions = C.consolidate(conn, root, config=config, **consolidate_kwargs)
    assert len(calls) == 1
    call = calls[0]
    assert call["kwargs"]["config"] is config
    previews = {
        path.name: path.read_text(encoding="utf-8").replace(str(root), ROOT_PLACEHOLDER)
        for path in sorted((root / ".suggestions").glob("*.md"))
    }
    rows = [
        list(row) for row in conn.execute(
            "SELECT kind, target_path, body, rationale, source_sessions, sug_file, base_sha256, "
            "target_existed, status FROM suggestions ORDER BY id"
        )
    ]
    record = {
        "stage": call["stage"],
        "model": call["kwargs"].get("model"),
        "schema": call["schema"],
        "previews": previews,
        "rows": rows,
        "suggestions": [[s.kind, s.target_path, s.base_sha256, s.target_existed] for s in suggestions],
        "consumed": [row[0] for row in conn.execute(
            "SELECT distillation_key FROM consolidated_distillations ORDER BY distillation_key")],
    }
    return call["prompt"].replace(str(root), ROOT_PLACEHOLDER), record


@pytest.mark.parametrize("consolidate_kwargs", [{}, {"mode": "full"}])
def test_full_mode_prompt_schema_and_outputs_match_the_pre_subset_golden(
    tmp_path, monkeypatch, consolidate_kwargs
):
    prompt, record = _run_full_mode(tmp_path, monkeypatch, **consolidate_kwargs)
    record_text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if os.environ.get("DREAM_UPDATE_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_PROMPT.write_text(prompt, encoding="utf-8")
        GOLDEN_CALL.write_text(record_text, encoding="utf-8")
    assert prompt == GOLDEN_PROMPT.read_text(encoding="utf-8")
    assert record_text == GOLDEN_CALL.read_text(encoding="utf-8")


def test_full_mode_never_offers_patch_and_keeps_the_staleness_audit(tmp_path, monkeypatch):
    prompt, record = _run_full_mode(tmp_path, monkeypatch)
    assert record["schema"] is not C.CONSOLIDATE_PATCH_SCHEMA
    assert "patch" not in json.dumps(record["schema"])
    assert "# Staleness audit" in prompt


# ------------------------------------------------------------ 60-file store

KEYWORDS = [f"kw{i:02d}" for i in range(60)]


def _topic_name(i: int) -> str:
    return f"project_topic_{i:02d}.md"


def _store(tmp_path, *, sync=True, extra=None):
    root = tmp_path / "memory"
    root.mkdir()
    index = []
    for i, keyword in enumerate(KEYWORDS):
        name = _topic_name(i)
        (root / name).write_text(
            f"---\nname: topic-{i:02d}\ndescription: about {keyword}\nmetadata:\n  type: project\n---\n\n"
            f"Facts about {keyword} and nothing else. Body marker BODY-{i:02d}.\n",
            encoding="utf-8",
        )
        index.append(f"- [Topic {i:02d}]({name}) — hook {i:02d}")
    for name, body in (extra or {}).items():
        (root / name).write_text(body, encoding="utf-8")
    (root / "MEMORY.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    dream._migrate(conn)
    if sync:
        synchronize_recall_documents(conn, root, include_raw_transcripts=False, constitution_path=None)
    return root, conn


def _batch(*findings, field="projects", summary=""):
    notes = {"summary": summary, "user_facts": [], "preferences": [], "projects": [], "references": [],
             "decisions": [], "open_threads": []}
    notes[field] = list(findings)
    return [("s-1", "2026-10-18T09:00:00Z", notes, "key-s-1")]


def _names(paths):
    return [path.name for path in paths]


def _select(conn, root, batch, settings, **kwargs):
    return C.select_memory_subset(conn, root, batch, settings, constitution_path=None, **kwargs)


def test_select_memory_subset_requires_the_constitution_path(tmp_path):
    root, conn = _store(tmp_path, sync=False)
    with pytest.raises(TypeError):
        C.select_memory_subset(conn, root, [], ConsolidateSettings(mode="subset"))


def test_subset_query_groups_are_capped(tmp_path, monkeypatch):
    root, conn = _store(tmp_path)
    calls = []
    real = C.rank_lexical
    monkeypatch.setattr(C, "rank_lexical", lambda conn, query, **kw: calls.append(query) or real(conn, query))
    findings = [" ".join(f"term{i}x{j}" for j in range(30)) for i in range(20)]
    _select(conn, root, _batch(*findings), ConsolidateSettings(mode="subset", subset_files=3))
    assert len(calls) == C.MAX_SUBSET_QUERIES == 8


def test_subset_always_contains_memory_md_first(tmp_path):
    root, conn = _store(tmp_path)
    for batch in (_batch("kw03 needs a new owner"), _batch("nothing matches here"), []):
        selected = _select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=5))
        assert _names(selected)[0] == "MEMORY.md"
        assert all(path.parent == root for path in selected)


def test_subset_takes_top_ranked_files_up_to_subset_files_without_counting_memory_md(tmp_path):
    root, conn = _store(tmp_path)
    mentioned = [3, 7, 11, 19, 23, 29, 31, 41]
    batch = _batch(*(f"Decision about kw{i:02d} rollout" for i in mentioned))
    selected = _select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=5))
    names = _names(selected)
    assert names[0] == "MEMORY.md"
    topics = names[1:]
    assert len(topics) == 5
    assert set(topics) <= {_topic_name(i) for i in mentioned}
    assert len(set(names)) == len(names)


def test_subset_query_is_not_cut_off_after_the_first_findings(tmp_path):
    root, conn = _store(tmp_path)
    filler = [f"Filler finding {' '.join(f'noise{i}{j}' for j in range(10))}" for i in range(6)]
    batch = _batch(*filler, "The kw57 migration finished.")
    selected = _select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=3))
    assert _topic_name(57) in _names(selected)


def test_subset_uses_preferences_projects_decisions_and_user_facts(tmp_path):
    root, conn = _store(tmp_path)
    for index, field in ((5, "preferences"), (6, "projects"), (8, "decisions"), (9, "user_facts")):
        selected = _select(
            conn, root, _batch(f"About kw{index:02d}", field=field), ConsolidateSettings(mode="subset", subset_files=2)
        )
        assert _topic_name(index) in _names(selected)


def test_slug_matches_use_stems_and_frontmatter_names_with_word_boundaries(tmp_path):
    extra = {
        "project_api.md": "---\nname: api\n---\n\nThe api.\n",
        "project_api_gateway.md": "---\nname: api-gateway\n---\n\nThe gateway.\n",
        "reference_dashboards.md": "---\nname: grafana-boards\n---\n\nBoards.\n",
        "reference_grafana_v2.md": "---\nname: grafana-boards-v2\n---\n\nBoards v2.\n",
    }
    root, conn = _store(tmp_path, sync=False, extra=extra)
    batch = _batch("See project_api_gateway.md for the routes.", "The Grafana-Boards page moved.",
                   field="decisions")
    selected = _names(_select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=1)))
    assert selected == ["MEMORY.md", "project_api_gateway.md", "reference_dashboards.md"]


def test_slug_mentioned_files_are_included_beyond_the_cap(tmp_path):
    root, conn = _store(tmp_path)
    batch = _batch("kw01 and kw02 changed", "project_topic_44.md and project_topic_45.md moved too")
    selected = _names(_select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=1)))
    topics = [name for name in selected if name.startswith("project_topic_")]
    assert len(topics) == 3
    assert {_topic_name(44), _topic_name(45)} <= set(topics)
    assert set(topics) - {_topic_name(44), _topic_name(45)} <= {_topic_name(1), _topic_name(2)}


def test_slug_matches_scan_every_finding_field_and_the_summary(tmp_path):
    root, conn = _store(tmp_path)
    for field in ("references", "open_threads"):
        selected = _names(_select(
            conn, root, _batch("project_topic_44.md", field=field), ConsolidateSettings(mode="subset", subset_files=1)
        ))
        assert _topic_name(44) in selected
    selected = _names(_select(
        conn, root, _batch(summary="touched topic-45 again"), ConsolidateSettings(mode="subset", subset_files=1)
    ))
    assert _topic_name(45) in selected


def test_missing_recall_index_falls_back_to_memory_md_and_slug_matches(tmp_path):
    root, conn = _store(tmp_path, sync=False)
    batch = _batch("kw10 changed", "project_topic_12.md was renamed")
    selected = _names(_select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=5)))
    assert selected == ["MEMORY.md", _topic_name(12)]


def test_broken_recall_tables_fall_back_without_crashing(tmp_path, capsys):
    root, conn = _store(tmp_path)
    conn.execute("DROP TABLE recall_documents_fts")
    conn.execute("DROP TABLE recall_documents")
    batch = _batch("kw10 changed", "project_topic_12.md was renamed")
    selected = _names(_select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=5)))
    assert selected == ["MEMORY.md", _topic_name(12)]
    assert "recall index unavailable" in capsys.readouterr().err


def test_stale_index_entries_for_deleted_or_excluded_files_are_skipped(tmp_path):
    root, conn = _store(tmp_path)
    (root / _topic_name(20)).unlink()
    (root / ".dream-retention.json").write_text(json.dumps({"memory_paths": [_topic_name(21)]}), encoding="utf-8")
    batch = _batch("kw20 and kw21 and kw22 changed", "project_topic_21.md too")
    selected = _names(_select(conn, root, batch, ConsolidateSettings(mode="subset", subset_files=5)))
    assert _topic_name(20) not in selected
    assert _topic_name(21) not in selected
    assert _topic_name(22) in selected


def test_dense_hook_is_off_without_an_embedder_and_fails_open(tmp_path):
    root, conn = _store(tmp_path)
    assert C._dense_for_subset(conn, "kw03", None) is None

    class BrokenEmbedder:
        fingerprint = "broken"
        remote_data_egress = False

        def embed(self, texts):
            raise RuntimeError("server down")

    assert C._dense_for_subset(conn, "kw03", BrokenEmbedder()) is None
    selected = _select(
        conn, root, _batch("kw03 moved"), ConsolidateSettings(mode="subset", subset_files=2), embedder=BrokenEmbedder()
    )
    assert _topic_name(3) in _names(selected)


class _MeaningEmbedder:
    """Deterministic vectors by topic: database words in English or Polish share an axis."""

    fingerprint = "meaning-v1"
    remote_data_egress = False

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0, 0.0] if any(word in text.casefold() for word in ("postgresql", "baza", "danych"))
                else [0.0, 1.0, 0.0] for text in texts]


_DATABASE_FILE = "project_database.md"
_DATABASE_BODY = ("---\nname: project-database\ndescription: storage\nmetadata:\n  type: project\n---\n\n"
                  "The service keeps its records in PostgreSQL.\n")


def _meaning_store(tmp_path):
    from recall_documents import embed_recall_documents
    root, conn = _store(tmp_path, extra={_DATABASE_FILE: _DATABASE_BODY})
    embedder = _MeaningEmbedder()
    assert embed_recall_documents(conn, embedder, include_raw_transcripts=False).error is None
    embedder.calls.clear()
    return root, conn, embedder


def test_dense_candidates_bring_a_file_reachable_only_by_meaning_into_the_subset(tmp_path):
    root, conn, embedder = _meaning_store(tmp_path)
    batch = _batch("baza danych przeniesiona na nowy serwer")
    settings = ConsolidateSettings(mode="subset", subset_files=2)
    assert _names(_select(conn, root, batch, settings)) == ["MEMORY.md"]
    selected = _names(_select(conn, root, batch, settings, embedder=embedder))
    assert selected[:2] == ["MEMORY.md", _DATABASE_FILE]
    assert embedder.calls == [["baza danych przeniesiona na nowy serwer"]]


def test_failing_embedder_keeps_the_lexical_subset_and_is_tried_once(tmp_path, capsys):
    root, conn, _embedder = _meaning_store(tmp_path)

    class Down(_MeaningEmbedder):
        def embed(self, texts):
            self.calls.append(list(texts))
            from recall_adapters import EmbedderUnavailable
            raise EmbedderUnavailable("refused")

    findings = [" ".join(f"kw{i:02d}" for i in range(start, start + 20)) for start in (0, 20, 40)]
    settings = ConsolidateSettings(mode="subset", subset_files=5)
    lexical = _names(_select(conn, root, _batch(*findings), settings))
    capsys.readouterr()
    down = Down()
    assert _names(_select(conn, root, _batch(*findings), settings, embedder=down)) == lexical
    assert len(down.calls) == 1
    err = [line for line in capsys.readouterr().err.splitlines() if line]
    assert len(err) == 1 and "dense candidates unavailable" in err[0]


def _subset_cli(tmp_path, monkeypatch, config_lines):
    import test_recall_embedder_http as http
    root, conn = _store(tmp_path, extra={_DATABASE_FILE: _DATABASE_BODY})
    conn.close()
    db_path = tmp_path / "dream.db"
    write = dream.open_db(db_path)
    synchronize_recall_documents(write, root, include_raw_transcripts=False, constitution_path=None)
    _distill(write, "s-1", {"projects": ["baza danych przeniesiona"]})
    write.close()
    path = tmp_path / "config.toml"
    path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    config = load_config(path, overrides={"storage": {"db_path": str(db_path), "memory_root": str(root)}})
    prompts = []
    monkeypatch.setattr(C, "generate", lambda stage, prompt, schema, **kw: prompts.append(prompt) or SimpleNamespace(
        output={"suggestions": []}, provider="fake", model="fake"))
    args = argparse.Namespace(db=db_path, memory=str(root), since=None, model=None, max_sessions=200, config=config)
    return args, prompts, http


def _embedder_lines(url):
    from test_recall_embedder_http import DIMENSION, MODEL
    return ["[recall.embedder]", "enabled = true", 'type = "openai-compatible"', f'base_url = "{url}"',
            f'model = "{MODEL}"', f"dimension = {DIMENSION}"]


def test_nightly_subset_consolidation_uses_the_configured_embedder(tmp_path, monkeypatch, capsys):
    from test_recall_embedder_http import FakeEmbeddingServer, make_embedder
    from recall_documents import embed_recall_documents
    server = FakeEmbeddingServer().start()
    try:
        args, prompts, _http = _subset_cli(tmp_path, monkeypatch, _embedder_lines(server.url) + ["[consolidate]", 'mode = "subset"', "subset_files = 1"])
        write = dream.open_db(args.db)
        embed_recall_documents(write, make_embedder(server.url), include_raw_transcripts=False)
        write.close()
        server.requests.clear()
        assert dream.cmd_consolidate(args) == 0
        assert server.posted_inputs() == ["baza danych przeniesiona"]
        assert "The service keeps its records in PostgreSQL." in prompts[0]
    finally:
        server.stop()


def test_nightly_subset_consolidation_with_a_misconfigured_embedder_ranks_lexically(tmp_path, monkeypatch, capsys):
    args, prompts, _http = _subset_cli(tmp_path, monkeypatch, _embedder_lines("http://192.0.2.10:8081") + ["[consolidate]", 'mode = "subset"'])
    assert dream.cmd_consolidate(args) == 0
    err = [line for line in capsys.readouterr().err.splitlines() if "embedder" in line]
    assert len(err) == 1 and "remote_data_egress" in err[0]
    assert len(prompts) == 1


def test_full_mode_consolidation_never_builds_the_embedder(tmp_path, monkeypatch, capsys):
    from test_recall_embedder_http import FakeEmbeddingServer
    server = FakeEmbeddingServer().start()
    try:
        args, prompts, _http = _subset_cli(tmp_path, monkeypatch, _embedder_lines(server.url))
        assert dream.cmd_consolidate(args) == 0
        assert server.requests == []
    finally:
        server.stop()


# ------------------------------------------------------------ subset consolidation


def _distill(conn, sid, notes):
    conn.execute(
        "INSERT INTO sessions(session_id, source, external_session_id, project_slug, jsonl_path, "
        "jsonl_mtime, ended_at, total_chars) VALUES (?, 'codex', ?, '-home-user', '/p.jsonl', 1, "
        "'2026-10-18T09:00:00Z', 1000)",
        (sid, sid),
    )
    full = {"summary": "", "user_facts": [], "preferences": [], "projects": [], "references": [],
            "decisions": [], "open_threads": []}
    full.update(notes)
    conn.execute(
        "INSERT INTO distilled(session_id, distillation_key, notes_json, summary) VALUES (?, ?, ?, '')",
        (sid, f"key-{sid}", json.dumps(full)),
    )
    conn.commit()


def _run_subset(tmp_path, monkeypatch, output, *, notes=None, subset_files=3):
    _fixed_clock(monkeypatch)
    root, conn = _store(tmp_path)
    _distill(conn, "s-1", notes if notes is not None else {"projects": ["kw04 and kw05 were reorganised"]})
    config = _config(tmp_path, root, mode="subset", subset_files=subset_files)
    calls = []

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append({"stage": stage, "prompt": prompt, "schema": schema})
        return SimpleNamespace(output=output, provider="fake", model="fake-model")

    monkeypatch.setattr(C, "generate", fake_generate)
    suggestions = C.consolidate(conn, root, config=config)
    return root, conn, calls, suggestions


def _suggestion(kind, target, body="", rationale="because", find="", sessions=("s-1",)):
    return {"kind": kind, "target_path": target, "find": find, "body": body,
            "rationale": rationale, "source_sessions": list(sessions)}


def test_subset_prompt_shows_selected_files_and_lists_the_rest_by_index_line(tmp_path, monkeypatch):
    root, conn, calls, _ = _run_subset(tmp_path, monkeypatch, {"suggestions": []})
    assert len(calls) == 1
    call = calls[0]
    prompt = call["prompt"]
    assert call["stage"] == "consolidate"
    assert call["schema"] is C.CONSOLIDATE_PATCH_SCHEMA
    assert "### FILE: MEMORY.md" in prompt
    assert "### FILE: project_topic_04.md" in prompt and "BODY-04" in prompt
    assert "### FILE: project_topic_05.md" in prompt and "BODY-05" in prompt
    assert "BODY-30" not in prompt
    assert "### FILE: project_topic_30.md" not in prompt
    other = prompt.split("## Other memory files", 1)[1].split("## New distilled notes", 1)[0]
    assert "- [Topic 30](project_topic_30.md) — hook 30" in other
    assert "(project_topic_04.md)" not in other
    assert "larger than" in prompt
    assert "61 files" in prompt
    assert "# Staleness audit" not in prompt
    assert "Do not propose `remove`" in prompt
    assert "`patch`" in prompt
    assert "{{" not in prompt
    assert "2026-10-19" in prompt


def test_subset_prompt_lists_unindexed_files_by_name(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch)
    root, conn = _store(tmp_path, extra={"reference_orphan.md": "---\nname: orphan\n---\n\nUnindexed.\n"})
    _distill(conn, "s-1", {"projects": ["kw04 moved"]})
    calls = []
    monkeypatch.setattr(C, "generate", lambda stage, prompt, schema, **kw: calls.append(prompt) or SimpleNamespace(
        output={"suggestions": []}, provider="fake", model="fake"))
    C.consolidate(conn, root, config=_config(tmp_path, root, mode="subset", subset_files=1))
    other = calls[0].split("## Other memory files", 1)[1]
    assert "reference_orphan.md" in other
    assert "Unindexed." not in calls[0]


def test_subset_mode_with_an_empty_batch_makes_no_provider_call(tmp_path, monkeypatch):
    root, conn = _store(tmp_path)
    calls = []
    monkeypatch.setattr(C, "generate", lambda *a, **k: calls.append(1))
    assert C.consolidate(conn, root, config=_config(tmp_path, root, mode="subset")) == []
    assert calls == []
    assert conn.execute("SELECT COUNT(*) FROM consolidation_runs").fetchone()[0] == 0


def _appended(root, name, addition):
    current = (root / name).read_text(encoding="utf-8")
    return current.rstrip("\n") + "\n\n" + addition


def test_new_on_an_existing_unshown_target_becomes_an_append_update(tmp_path, monkeypatch):
    output = {"suggestions": [_suggestion("new", "project_topic_30.md", "---\nname: topic-30\n---\n\nRewritten.\n",
                                          "fresh topic")]}
    root, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    [s] = suggestions
    assert s.kind == "update"
    assert s.target_existed is True
    assert s.base_sha256 == _sha(root / "project_topic_30.md")
    assert s.body == _appended(root, "project_topic_30.md", "Rewritten.\n")
    assert s.body.startswith((root / "project_topic_30.md").read_text(encoding="utf-8").rstrip("\n"))
    assert s.rationale.startswith("fresh topic")
    assert "converted from `new`" in s.rationale
    assert "appended: target not shown to the model" in s.rationale
    row = conn.execute("SELECT kind, body, base_sha256, target_existed FROM suggestions").fetchone()
    assert row == ("update", s.body, _sha(root / "project_topic_30.md"), 1)


def test_new_on_an_existing_shown_target_becomes_a_plain_update(tmp_path, monkeypatch):
    body = "---\nname: topic-04\n---\n\nRewritten after reading it.\n"
    output = {"suggestions": [_suggestion("new", "project_topic_04.md", body, "fresh topic")]}
    root, _, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    [s] = suggestions
    assert (s.kind, s.body, s.base_sha256) == ("update", body, _sha(root / "project_topic_04.md"))
    assert "converted from `new`" in s.rationale
    assert "appended" not in s.rationale


def test_update_on_an_unshown_target_appends_and_never_drops_its_body(tmp_path, monkeypatch):
    output = {"suggestions": [
        _suggestion("update", "project_topic_32.md", "---\nname: topic-32\n---\n\nOnly new facts.\n"),
        _suggestion("update", "project_topic_33.md", "---\nname: topic-33\n---\n\n"),
    ]}
    root, _, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    [s] = suggestions
    assert s.target_path == "project_topic_32.md"
    assert s.body == _appended(root, "project_topic_32.md", "Only new facts.\n")
    assert "BODY-32" in s.body
    assert s.base_sha256 == _sha(root / "project_topic_32.md")
    assert "appended: target not shown to the model" in s.rationale


def test_subset_drops_suggestions_for_existing_files_that_are_not_eligible(tmp_path, monkeypatch, capsys):
    _fixed_clock(monkeypatch)
    root, conn = _store(tmp_path, extra={"project_private.md": "---\nname: private\n---\n\nPRIVATE-MARKER notes.\n"})
    (root / ".dream-retention.json").write_text(json.dumps({"text_markers": ["PRIVATE-MARKER"]}), encoding="utf-8")
    _distill(conn, "s-1", {"projects": ["kw04 moved"]})
    monkeypatch.setattr(C, "generate", lambda *a, **k: SimpleNamespace(output={"suggestions": [
        _suggestion("update", "project_private.md", "---\nname: private\n---\n\nHarmless addition.\n"),
    ]}, provider="fake", model="fake"))
    assert C.consolidate(conn, root, config=_config(tmp_path, root, mode="subset", subset_files=1)) == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0
    assert "dropped update project_private.md" in capsys.readouterr().out


def test_new_on_an_existing_target_is_matched_by_its_normalized_name(tmp_path, monkeypatch):
    output = {"suggestions": [_suggestion("new", "./project_topic_30.md", "---\nname: topic-30\n---\n\nNew.\n")]}
    root, _, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert [(s.kind, s.base_sha256) for s in suggestions] == [("update", _sha(root / "project_topic_30.md"))]


def test_new_on_a_missing_target_stays_new(tmp_path, monkeypatch):
    output = {"suggestions": [_suggestion("new", "project_brand_new.md", "---\nname: brand-new\n---\n\nNew.\n")]}
    _, _, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert [(s.kind, s.target_existed, s.base_sha256) for s in suggestions] == [("new", False, None)]


def test_subset_drops_every_remove_and_logs_it(tmp_path, monkeypatch, capsys):
    output = {"suggestions": [
        _suggestion("remove", "project_topic_04.md", rationale="DEPRECATED"),
        _suggestion("remove", "project_topic_30.md", rationale="DEPRECATED"),
    ]}
    _, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert suggestions == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0
    out = capsys.readouterr().out
    assert "dropped remove project_topic_04.md" in out
    assert "dropped remove project_topic_30.md" in out


def test_subset_drops_patches_for_files_not_shown_and_keeps_shown_updates(tmp_path, monkeypatch, capsys):
    shown_body = "---\nname: topic-05\n---\n\nSeen and rewritten.\n"
    output = {"suggestions": [
        _suggestion("patch", "project_topic_31.md", "new text", find="Facts about kw31"),
        _suggestion("update", "project_topic_32.md", "---\nname: topic-32\n---\n\nOnly new facts.\n"),
        _suggestion("update", "project_topic_05.md", shown_body),
        _suggestion("patch", "project_topic_04.md", "Facts about kw04, now owned by ops", find="Facts about kw04"),
    ]}
    _, _, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    by_target = {s.target_path: s for s in suggestions}
    assert set(by_target) == {"project_topic_32.md", "project_topic_05.md", "project_topic_04.md"}
    assert "appended: target not shown to the model" in by_target["project_topic_32.md"].rationale
    assert (by_target["project_topic_05.md"].body, by_target["project_topic_05.md"].rationale) == (shown_body, "because")
    assert by_target["project_topic_04.md"].kind == "patch"
    assert "dropped patch project_topic_31.md" in capsys.readouterr().out


@pytest.mark.parametrize("find", [
    "Facts about kw04 and nothing else. Body marker BODY-04.\n",
    "---\nname: topic-04\ndescription: about kw04\nmetadata:\n  type: project\n---\n\n"
    "Facts about kw04 and nothing else. Body marker BODY-04.\n",
])
def test_subset_drops_a_patch_that_would_empty_a_file(tmp_path, monkeypatch, capsys, find):
    output = {"suggestions": [_suggestion("patch", "project_topic_04.md", "", "delete obsolete file", find=find)]}
    root, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert suggestions == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0
    out = capsys.readouterr().out
    assert "dropped patch project_topic_04.md" in out and "would empty the file" in out


def test_subset_never_persists_retention_excluded_find_text(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch)
    root, conn = _store(tmp_path, extra={"project_topic_04.md": (
        "---\nname: topic-04\n---\n\nFacts about kw04. Old SECRET-X note.\n")})
    (root / ".dream-retention.json").write_text(json.dumps({"text_markers": ["SECRET-Y"]}), encoding="utf-8")
    _distill(conn, "s-1", {"projects": ["kw04 moved"]})
    monkeypatch.setattr(C, "generate", lambda *a, **k: SimpleNamespace(output={"suggestions": [
        _suggestion("patch", "project_topic_04.md", "Old note.", find="Old SECRET-Y note."),
        _suggestion("update", "project_topic_04.md", "---\nname: topic-04\n---\n\nFine.\n", find="SECRET-Y"),
    ]}, provider="fake", model="fake"))
    suggestions = C.consolidate(conn, root, config=_config(tmp_path, root, mode="subset", subset_files=2))
    assert [(s.kind, s.find) for s in suggestions] == [("update", None)]
    stored = json.dumps(conn.execute("SELECT * FROM suggestions").fetchall())
    previews = "".join(path.read_text(encoding="utf-8") for path in root.glob(".suggestions/*.md"))
    assert "SECRET-Y" not in stored and "SECRET-Y" not in previews


def test_subset_base_sha_is_the_hash_of_the_bytes_the_model_saw(tmp_path, monkeypatch, capsys):
    real_select = C.select_memory_subset

    def select_then_edit(*args, **kwargs):
        (args[1] / "project_topic_04.md").write_text("---\nname: topic-04\n---\n\nEdited by hand.\n", encoding="utf-8")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(C, "select_memory_subset", select_then_edit)
    output = {"suggestions": [
        _suggestion("update", "project_topic_04.md", "---\nname: topic-04\n---\n\nModel rewrite.\n")]}
    root, conn, calls, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert "BODY-04" in calls[0]["prompt"]
    [s] = suggestions
    assert s.base_sha256 != _sha(root / "project_topic_04.md")
    conn.close()
    db_path = tmp_path / "accept.db"
    write = dream.open_db(db_path)
    write.execute(
        "INSERT INTO suggestions(kind, target_path, body, rationale, status, base_sha256, target_existed) "
        "VALUES (?, ?, ?, ?, 'pending', ?, 1)", (s.kind, s.target_path, s.body, s.rationale, s.base_sha256))
    write.commit()
    write.close()
    args = argparse.Namespace(db=db_path, memory=str(root), suggestions_cmd="accept", id=1,
                              config=_config(tmp_path, root))
    assert dream.cmd_suggestions(args) == 3
    assert "conflict #1" in capsys.readouterr().out
    assert "Edited by hand." in (root / "project_topic_04.md").read_text(encoding="utf-8")


def test_preview_names_never_collide_with_an_existing_preview(tmp_path, monkeypatch):
    _fixed_clock(monkeypatch)
    root, conn = _store(tmp_path)
    _distill(conn, "s-1", {"projects": ["kw04 and kw05 were reorganised"]})
    sug_dir = root / ".suggestions"
    sug_dir.mkdir()
    taken = sug_dir / "2026-10-19-002-update-project-topic-04.md"
    taken.write_text("someone else's preview", encoding="utf-8")
    monkeypatch.setattr(C, "generate", lambda *a, **k: SimpleNamespace(output={"suggestions": [
        _suggestion("update", "project_topic_04.md", "---\nname: topic-04\n---\n\nA.\n"),
        _suggestion("update", "project_topic_05.md", "---\nname: topic-05\n---\n\nB.\n"),
    ]}, provider="fake", model="fake"))
    C.consolidate(conn, root, config=_config(tmp_path, root, mode="subset", subset_files=3))
    assert taken.read_text(encoding="utf-8") == "someone else's preview"
    names = [row[0] for row in conn.execute("SELECT sug_file FROM suggestions ORDER BY id")]
    assert names == ["2026-10-19-003-update-project-topic-04.md", "2026-10-19-004-update-project-topic-05.md"]
    assert all((sug_dir / name).exists() for name in names)


def test_subset_patch_is_persisted_with_find_text_and_a_unified_diff_preview(tmp_path, monkeypatch):
    output = {"suggestions": [
        _suggestion("patch", "project_topic_04.md", "Facts about kw04, now owned by ops", find="Facts about kw04"),
    ]}
    root, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    [s] = suggestions
    assert (s.kind, s.find, s.base_sha256, s.target_existed) == (
        "patch", "Facts about kw04", _sha(root / "project_topic_04.md"), True)
    row = conn.execute("SELECT kind, find_text, body, sug_file FROM suggestions").fetchone()
    assert row[:3] == ("patch", "Facts about kw04", "Facts about kw04, now owned by ops")
    preview = (root / ".suggestions" / row[3]).read_text(encoding="utf-8")
    assert "kind: patch" in preview
    assert "--- a/project_topic_04.md" in preview
    assert "+++ b/project_topic_04.md" in preview
    assert "-Facts about kw04 and nothing else. Body marker BODY-04." in preview
    assert "+Facts about kw04, now owned by ops and nothing else. Body marker BODY-04." in preview


def test_subset_patch_preview_explains_an_ambiguous_find(tmp_path, monkeypatch):
    output = {"suggestions": [_suggestion("patch", "project_topic_04.md", "X", find="---")]}
    root, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert [s.kind for s in suggestions] == ["patch"]
    sug_file = conn.execute("SELECT sug_file FROM suggestions").fetchone()[0]
    preview = (root / ".suggestions" / sug_file).read_text(encoding="utf-8")
    assert "cannot apply" in preview
    assert "2 places" in preview


@pytest.mark.parametrize("bad", [
    _suggestion("patch", "project_topic_04.md", "x", find=""),
    _suggestion("patch", "project_missing.md", "x", find="Facts"),
    _suggestion("patch", "MEMORY.md", "- [Topic 04](project_topic_04.md) — renamed", find="- [Topic 04]"),
])
def test_subset_drops_unusable_patches(tmp_path, monkeypatch, bad):
    _, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, {"suggestions": [bad]})
    assert suggestions == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0


def test_subset_non_patch_suggestions_store_no_find_text(tmp_path, monkeypatch):
    output = {"suggestions": [
        _suggestion("update", "project_topic_04.md", "---\nname: topic-04\n---\n\nNew.\n", find="ignored"),
        _suggestion("index", "MEMORY.md", "- [New](project_new.md) — new\n"),
    ]}
    _, conn, _, suggestions = _run_subset(tmp_path, monkeypatch, output)
    assert [s.find for s in suggestions] == [None, None]
    assert [row[0] for row in conn.execute("SELECT find_text FROM suggestions ORDER BY id")] == [None, None]


def test_subset_consumes_the_batch_like_full_mode(tmp_path, monkeypatch):
    _, conn, _, _ = _run_subset(tmp_path, monkeypatch, {"suggestions": []})
    assert [row[0] for row in conn.execute("SELECT distillation_key FROM consolidated_distillations")] == ["key-s-1"]


def test_patch_schema_extends_the_full_schema_with_patch_and_a_required_find():
    item = C.CONSOLIDATE_PATCH_SCHEMA["properties"]["suggestions"]["items"]
    full_item = C.CONSOLIDATE_SCHEMA["properties"]["suggestions"]["items"]
    assert item["properties"]["kind"]["enum"] == ["new", "update", "remove", "index", "patch"]
    assert item["properties"]["find"] == {"type": "string"}
    assert set(item["required"]) == set(item["properties"])
    assert item["additionalProperties"] is False
    assert {k: v for k, v in item["properties"].items() if k not in {"kind", "find"}} == {
        k: v for k, v in full_item["properties"].items() if k != "kind"}
    assert full_item["properties"]["kind"]["enum"] == ["new", "update", "remove", "index"]


# ------------------------------------------------------------ configuration and CLI


def test_consolidate_settings_default_to_full_mode(tmp_path):
    config = load_config(tmp_path / "missing-config.toml")
    assert config.consolidate_settings() == ConsolidateSettings(mode="full", subset_files=15)


@pytest.mark.parametrize("table, message", [
    ({"mode": "partial"}, "consolidate.mode"),
    ({"subset_files": 0}, "consolidate.subset_files"),
    ({"subset_files": True}, "consolidate.subset_files"),
    ({"subset_files": "5"}, "consolidate.subset_files"),
    ({"unknown": 1}, "unknown consolidate options"),
])
def test_consolidate_settings_are_validated(tmp_path, table, message):
    with pytest.raises(ConfigError, match=message):
        load_config(tmp_path / "missing-config.toml", overrides={"consolidate": table})


def test_cli_mode_flag_overrides_the_configured_mode(tmp_path, monkeypatch):
    root, conn = _store(tmp_path)
    conn.close()
    db_path = tmp_path / "dream.db"
    write = dream.open_db(db_path)
    _distill(write, "s-1", {"projects": ["kw04 moved"]})
    write.close()
    config = _config(tmp_path, root)
    seen = []
    monkeypatch.setattr(C, "generate", lambda stage, prompt, schema, **kw: seen.append(schema) or SimpleNamespace(
        output={"suggestions": []}, provider="fake", model="fake"))
    args = argparse.Namespace(db=db_path, memory=str(root), since=None, model=None, max_sessions=200,
                              config=config, mode="subset")
    assert dream.cmd_consolidate(args) == 0
    assert seen == [C.CONSOLIDATE_PATCH_SCHEMA]


def test_cli_rejects_an_unknown_mode():
    with pytest.raises(SystemExit):
        dream.main(["consolidate", "--mode", "partial"])
