import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dream
import distill
import consolidate
from config import load_config
from curate import build_curation_prompt
from recall_context import run_context
from recall_documents import synchronize_recall_documents
from recall_types import RecallQuery
from retention_cleanup import apply_retention
from retention_policy import POLICY_FILE, load_retention_policy


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    conn = dream.open_db(tmp_path / "dream.db")
    conn.execute("INSERT INTO sessions(session_id,project_slug,jsonl_path,jsonl_mtime,total_chars) "
                 "VALUES ('private-session','private-project','/fixture',1,1000)")
    conn.execute("INSERT INTO distilled(session_id,notes_json,summary,distillation_key) "
                 "VALUES ('private-session',?,?,'private-key')",
                 (json.dumps({"summary": "retired-marker private detail", "projects": ["retired-marker"]}), "retired-marker"))
    conn.commit()
    (root / "private.md").write_text("retired-marker private detail")
    (root / "ordinary.md").write_text("ordinary useful preference")
    (root / "MEMORY.md").write_text("- [Private](private.md)\n- [Ordinary](ordinary.md)\n")
    synchronize_recall_documents(conn, root, include_raw_transcripts=False)
    (root / POLICY_FILE).write_text(json.dumps({
        "instructions": "Do not retain the retired project, including paraphrases.",
        "memory_paths": ["private.md"], "project_slugs": ["private-project"],
        "text_markers": ["retired-marker"],
    }))
    yield conn, root, tmp_path
    conn.close()


def test_existing_rows_are_not_recalled_before_cleanup(setup):
    conn, root, _ = setup
    cfg = load_config()
    for event in ("session-start:startup", "prompt"):
        query = RecallQuery("retired-marker", "test", event, str(root), (), 6000, frozenset(), False)
        result = run_context(conn, query, cfg.recall_settings(), memory_root=root)
        assert "retired-marker" not in result.rendered_context
        query = RecallQuery("ordinary", "test", event, str(root), (), 6000, frozenset(), False)
        result = run_context(conn, query, cfg.recall_settings(), memory_root=root)
        assert "ordinary useful preference" in result.rendered_context


def test_cleanup_and_reindex_preserve_unrelated_memories(setup):
    conn, root, _ = setup
    before = (root / "ordinary.md").read_bytes()
    preview = apply_retention(conn, root)
    assert preview["memory_files"] == ["private.md"]
    assert preview["distilled_records"] == 1
    assert (root / "private.md").exists()
    result = apply_retention(conn, root, apply=True)
    assert (Path(result["backup"]) / "affected-records.json").exists()
    assert not (root / "private.md").exists()
    assert "private.md" not in (root / "MEMORY.md").read_text()
    assert (root / "ordinary.md").read_bytes() == before
    assert "retired-marker" not in conn.execute("SELECT notes_json FROM distilled").fetchone()[0]
    assert conn.execute("SELECT count(*) FROM recall_documents WHERE text LIKE '%retired-marker%'").fetchone()[0] == 0
    (root / "private.md").write_text("retired-marker recreated")
    synchronize_recall_documents(conn, root, include_raw_transcripts=False)
    assert conn.execute("SELECT count(*) FROM recall_documents WHERE text LIKE '%retired-marker%'").fetchone()[0] == 0


def test_excluded_project_never_calls_distillation_provider(setup, monkeypatch):
    conn, root, _ = setup
    conn.execute("INSERT INTO messages(session_id,seq,role,text) VALUES ('private-session',0,'user','project detail')")
    conn.commit()
    monkeypatch.setattr(distill, "generate", lambda *a, **kw: pytest.fail("excluded project reached provider"))
    assert distill.distill_session(conn, "private-session", memory_root=root) is None


def test_consolidation_context_and_write_filter(setup, monkeypatch):
    conn, root, _ = setup
    def generate(stage, prompt, schema, **kwargs):
        assert "Do not retain the retired project, including paraphrases." in prompt
        assert "private detail" not in prompt
        return SimpleNamespace(provider="fixture", model="fixture", output={"suggestions": [
            {"kind": "new", "target_path": "private.md", "body": "renamed detail", "source_sessions": [], "rationale": "old context"},
            {"kind": "new", "target_path": "renamed.md", "body": "retired-marker detail", "source_sessions": [], "rationale": "old context"},
            {"kind": "new", "target_path": "allowed.md", "body": "use short dates", "source_sessions": [], "rationale": "preference"},
        ]})
    monkeypatch.setattr(consolidate, "generate", generate)
    suggestions = consolidate.consolidate(conn, root)
    assert [s.target_path for s in suggestions] == ["allowed.md"]
    assert conn.execute("SELECT count(*) FROM suggestions").fetchone()[0] == 1
    assert "Do not retain the retired project" in build_curation_prompt([], root)


def test_pending_suggestion_cannot_restore_excluded_memory(setup):
    conn, root, _ = setup
    conn.execute("INSERT INTO suggestions(kind,target_path,body) VALUES ('new','private.md','new words')")
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    (root / "private.md").unlink()
    assert not dream._apply_suggestion(conn, root, root / ".suggestions", sid, "new", "private.md", "new words", None)
    assert not (root / "private.md").exists()
    assert conn.execute("SELECT status FROM suggestions WHERE id=?", (sid,)).fetchone()[0] == "rejected"


def test_invalid_policy_does_not_silently_allow_recall(setup):
    _, root, _ = setup
    (root / POLICY_FILE).write_text('{"text_markers":[""]}')
    with pytest.raises(ValueError):
        load_retention_policy(root)


def test_model_cannot_replace_retention_policy(setup):
    conn, root, _ = setup
    before = (root / POLICY_FILE).read_bytes()
    conn.execute("INSERT INTO suggestions(kind,target_path,body) VALUES ('update',?,'{}')", (POLICY_FILE,))
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    assert not dream._apply_suggestion(conn, root, root / ".suggestions", sid, "update", POLICY_FILE, "{}", None)
    assert (root / POLICY_FILE).read_bytes() == before


def test_distillation_receives_policy_and_drops_excluded_facts(setup, monkeypatch):
    conn, root, _ = setup
    conn.execute("UPDATE sessions SET project_slug='ordinary-project' WHERE session_id='private-session'")
    conn.execute("INSERT INTO messages(session_id,seq,role,text) VALUES ('private-session',0,'user','mixed topic')")
    conn.commit()
    def generate(stage, prompt, schema, **kwargs):
        assert "Do not retain the retired project, including paraphrases." in prompt
        return SimpleNamespace(output={"summary": "retired-marker report", "projects": ["retired-marker", "ordinary project"]},
            input_tokens=1, output_tokens=1, cache_creation_tokens=None, cache_read_tokens=None,
            total_cost_usd=None, provider="fixture", model="fixture", duration_ms=1, usage=None)
    monkeypatch.setattr(distill, "generate", generate)
    result = distill.distill_session(conn, "private-session", memory_root=root)
    assert result.notes == {"summary": "", "projects": ["ordinary project"]}
    assert "retired-marker" not in conn.execute("SELECT notes_json FROM distilled").fetchone()[0]


def test_cleanup_rejects_pending_suggestion_from_excluded_project(setup):
    conn, root, _ = setup
    conn.execute("INSERT INTO suggestions(kind,target_path,body,source_sessions) "
                 "VALUES ('new','neutral.md','use short dates','private-session')")
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    assert apply_retention(conn, root)["pending_suggestions"] == 1
    apply_retention(conn, root, apply=True)
    assert conn.execute("SELECT status FROM suggestions WHERE id=?", (sid,)).fetchone()[0] == "rejected"


def test_apply_rejects_pending_suggestion_from_excluded_project(setup):
    conn, root, _ = setup
    conn.execute("INSERT INTO suggestions(kind,target_path,body,source_sessions) "
                 "VALUES ('new','neutral.md','use short dates','other-session,private-session')")
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    assert not dream._apply_suggestion(conn, root, root / ".suggestions", sid,
                                       "new", "neutral.md", "use short dates", None)
    assert not (root / "neutral.md").exists()
    assert conn.execute("SELECT status FROM suggestions WHERE id=?", (sid,)).fetchone()[0] == "rejected"


@pytest.mark.parametrize("policy_path,target_path", [
    ("./private.md", "private.md"), ("private.md", "./private.md"),
])
def test_equivalent_paths_cannot_restore_excluded_memory(setup, policy_path, target_path):
    conn, root, _ = setup
    (root / POLICY_FILE).write_text(json.dumps({"memory_paths": [policy_path]}))
    before = (root / "private.md").read_bytes()
    conn.execute("INSERT INTO suggestions(kind,target_path,body) VALUES ('update',?,'new words')", (target_path,))
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    assert not dream._apply_suggestion(conn, root, root / ".suggestions", sid,
                                       "update", target_path, "new words", None)
    assert (root / "private.md").read_bytes() == before


def test_cleanup_normalizes_policy_paths_for_index_removal(setup):
    conn, root, _ = setup
    (root / POLICY_FILE).write_text(json.dumps({"memory_paths": ["./private.md"]}))
    apply_retention(conn, root, apply=True)
    assert "private.md" not in (root / "MEMORY.md").read_text()


@pytest.mark.parametrize("alias", ["absolute", "parent", "symlink"])
def test_resolved_paths_cannot_restore_excluded_memory(setup, alias):
    conn, root, _ = setup
    (root / "sub").mkdir()
    (root / "alias.md").symlink_to(root / "private.md")
    target = {"absolute": str(root / "private.md"), "parent": "sub/../private.md", "symlink": "alias.md"}[alias]
    before = (root / "private.md").read_bytes()
    conn.execute("INSERT INTO suggestions(kind,target_path,body) VALUES ('update',?,'new words')", (target,))
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    assert not dream._apply_suggestion(conn, root, root / ".suggestions", sid,
                                       "update", target, "new words", None)
    assert (root / "private.md").read_bytes() == before


def test_cleanup_rejects_pending_suggestion_for_resolved_excluded_path(setup):
    conn, root, _ = setup
    conn.execute("INSERT INTO suggestions(kind,target_path,body) VALUES ('update',?,'new words')", (str(root / "private.md"),))
    conn.commit()
    assert apply_retention(conn, root)["pending_suggestions"] == 1


def test_consolidation_filters_resolved_excluded_paths(setup, monkeypatch):
    conn, root, _ = setup
    monkeypatch.setattr(consolidate, "generate", lambda *a, **kw: SimpleNamespace(
        provider="fixture", model="fixture", output={"suggestions": [{
            "kind": "update", "target_path": str(root / "private.md"), "body": "new words",
            "rationale": "preference", "source_sessions": [],
        }]}))
    assert consolidate.consolidate(conn, root) == []


@pytest.mark.parametrize("stage", ["recall", "index", "consolidation-context"])
def test_excluded_memory_symlink_is_not_read(setup, stage):
    conn, root, _ = setup
    (root / "alias.md").symlink_to(root / "private.md")
    (root / POLICY_FILE).write_text("{}")
    synchronize_recall_documents(conn, root, include_raw_transcripts=False)
    (root / POLICY_FILE).write_text(json.dumps({"memory_paths": ["private.md"]}))
    if stage == "recall":
        query = RecallQuery("retired-marker", "test", "prompt", str(root), (), 6000, frozenset(), False)
        result = run_context(conn, query, load_config().recall_settings(), memory_root=root)
        assert not any(c.document.source_path == "alias.md" for c in result.selected)
    elif stage == "index":
        synchronize_recall_documents(conn, root, include_raw_transcripts=False)
        assert conn.execute("SELECT count(*) FROM recall_documents WHERE source_path='alias.md'").fetchone()[0] == 0
    else:
        assert "retired-marker private detail" not in consolidate.read_current_memory(root)


@pytest.mark.parametrize("resource", ["index", "topic", "distilled"])
def test_cleanup_preserves_changes_made_after_snapshot(setup, monkeypatch, resource):
    conn, root, _ = setup
    conn.execute("UPDATE sessions SET project_slug='ordinary-project'")
    conn.commit()
    real_dump = json.dump
    fresh_notes = json.dumps({"summary": "new ordinary fact", "projects": ["new ordinary project"]})

    def write_backup_then_edit(*args, **kwargs):
        real_dump(*args, **kwargs)
        if resource == "index":
            with (root / "MEMORY.md").open("a") as handle:
                handle.write("- [New](new.md)\n")
        elif resource == "topic":
            (root / "private.md").write_text("new user edit")
        else:
            conn.execute("UPDATE distilled SET notes_json=?,summary='new ordinary fact'", (fresh_notes,))
            conn.commit()

    monkeypatch.setattr("retention_cleanup.json.dump", write_backup_then_edit)
    result = apply_retention(conn, root, apply=True)
    if resource == "index":
        assert "[New](new.md)" in (root / "MEMORY.md").read_text()
        assert result["deferred_files"] == ["MEMORY.md"]
    elif resource == "topic":
        assert (root / "private.md").read_text() == "new user edit"
        assert result["deferred_files"] == ["private.md"]
    else:
        assert conn.execute("SELECT notes_json,summary FROM distilled").fetchone() == (fresh_notes, "new ordinary fact")
        assert result["deferred_sessions"] == ["private-session"]
