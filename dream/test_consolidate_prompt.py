"""Tests for the consolidate prompt templates: the staleness audit and recall usage annotations."""
import json
import os
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import consolidate as C
import dream
from config import load_config
from consolidate import _load_prompt
from recall_documents import stable_document_id, synchronize_recall_documents


def test_prompt_has_staleness_audit_section():
    assert "# Staleness audit" in _load_prompt()


def test_prompt_names_closure_markers():
    text = _load_prompt()
    assert "ODRZUCONE" in text
    assert "ARCHIWALNE" in text


def test_prompt_tells_model_not_to_judge_age():
    text = _load_prompt()
    assert "Do NOT use file age" in text


def test_prompt_substitution_leaves_no_placeholders():
    filled = (
        _load_prompt()
        .replace("{{today}}", "2026-07-12")
        .replace("{{memory_root}}", "/tmp/memory")
        .replace("{{session_count}}", "0")
        .replace("{{current_memory}}", "(empty)")
        .replace("{{distilled_batch}}", "(empty)")
    )
    assert "{{" not in filled


# ------------------------------------------------------------ recall usage annotations

FILE_BLOCK = re.compile(r"^### FILE: (?P<name>.+)\n(?P<next>.*)$", re.MULTILINE)


def _usage_store(tmp_path, *, since="2026-07-01T08:00:00Z"):
    root = tmp_path / "memory"
    root.mkdir()
    names = ["project_alpha.md", "project_beta.md", "project_gamma.md"]
    for name in names:
        (root / name).write_text(
            f"---\nname: {name[:-3]}\nmetadata:\n  type: project\n---\n\n{name[8:-3]} facts about kwalpha.\n",
            encoding="utf-8",
        )
    (root / "MEMORY.md").write_text("".join(f"- [{n}]({n}) — hook\n" for n in names), encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.executescript((Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))
    dream._migrate(conn)
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, constitution_path=None)
    conn.execute(
        "INSERT INTO recall_usage VALUES (?, '2026-07-02T00:00:00Z', '2026-10-01T09:30:00Z', 3)",
        (stable_document_id("approved_memory", "memory:project_alpha.md"),),
    )
    if since:
        conn.execute("INSERT INTO meta(key, value) VALUES ('recall_usage_since', ?)", (since,))
    conn.execute(
        "INSERT INTO sessions(session_id, source, external_session_id, project_slug, jsonl_path, jsonl_mtime, "
        "ended_at, total_chars) VALUES ('s-1', 'codex', 's-1', '-home-user', '/p.jsonl', 1, '2026-10-18T09:00:00Z', 1)"
    )
    notes = {"summary": "", "user_facts": [], "preferences": [], "projects": ["kwalpha project_beta changed"],
             "references": [], "decisions": [], "open_threads": []}
    conn.execute("INSERT INTO distilled(session_id, distillation_key, notes_json, summary) VALUES "
                 "('s-1', 'key-s-1', ?, '')", (json.dumps(notes),))
    conn.commit()
    return root, conn


def _fake_provider(monkeypatch, output=None):
    calls = []

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append(prompt)
        return SimpleNamespace(output=output or {"suggestions": []}, provider="fake", model="fake-model")

    monkeypatch.setattr(C, "generate", fake_generate)
    return calls


def _config(tmp_path, root, mode):
    return load_config(tmp_path / "missing.toml", overrides={
        "storage": {"db_path": str(tmp_path / "dream.db"), "memory_root": str(root)},
        "consolidate": {"mode": mode}})


def _annotations(prompt):
    return {match["name"]: match["next"] for match in FILE_BLOCK.finditer(prompt)}


def test_subset_prompt_annotates_every_shown_file_with_its_recall_usage(tmp_path, monkeypatch):
    root, conn = _usage_store(tmp_path)
    calls = _fake_provider(monkeypatch)
    C.consolidate(conn, root, config=_config(tmp_path, root, "subset"))
    [prompt] = calls
    annotations = _annotations(prompt)
    assert annotations["project_alpha.md"] == "last recalled: 2026-10-01; recalls: 3"
    assert annotations["project_beta.md"] == "last recalled: never; recalls: 0"
    assert annotations["MEMORY.md"] == "last recalled: never; recalls: 0"
    assert "Recall usage observed since: 2026-07-01" in prompt
    assert "`last recalled:" in prompt and "never a reason on its own" in prompt
    assert "{{" not in prompt


def test_audit_prompt_annotates_every_file_and_keeps_the_age_rule(tmp_path, monkeypatch):
    root, conn = _usage_store(tmp_path, since=None)
    calls = _fake_provider(monkeypatch)
    C.run_audit(conn, root, config=_config(tmp_path, root, "subset"))
    [prompt] = calls
    assert _annotations(prompt) == {
        "project_alpha.md": "last recalled: 2026-10-01; recalls: 3",
        "project_beta.md": "last recalled: never; recalls: 0",
        "project_gamma.md": "last recalled: never; recalls: 0",
    }
    assert "Recall usage observed since: not observed yet" in prompt
    assert "Do NOT use file age" in prompt
    assert "never a reason on its own" in prompt
    assert "{{" not in prompt


def test_full_mode_prompt_carries_no_usage_annotations(tmp_path, monkeypatch):
    root, conn = _usage_store(tmp_path)
    calls = _fake_provider(monkeypatch)
    C.consolidate(conn, root, config=_config(tmp_path, root, "full"))
    [prompt] = calls
    assert "last recalled" not in prompt and "recalls:" not in prompt
    assert "Recall usage observed" not in prompt
    assert set(_annotations(prompt).values()) == {""}


def test_a_never_recalled_file_is_still_held_back_by_the_grace_period(tmp_path, monkeypatch):
    root, conn = _usage_store(tmp_path)
    old = (Path(root) / "project_gamma.md")
    stamp = old.stat().st_mtime - 40 * 86400
    os.utime(old, (stamp, stamp))
    remove = {"kind": "remove", "find": "", "body": "", "rationale": "DEPRECATED, never recalled",
              "source_sessions": []}
    calls = _fake_provider(monkeypatch, {"suggestions": [
        dict(remove, target_path="project_beta.md"), dict(remove, target_path="project_gamma.md")]})
    suggestions = C.run_audit(conn, root, config=_config(tmp_path, root, "subset"))
    assert len(calls) == 1
    # Usage changes neither gate: the fresh file is held back, the old one is only proposed.
    assert [(s.kind, s.target_path) for s in suggestions] == [("remove", "project_gamma.md")]
    assert (root / "project_beta.md").exists() and (root / "project_gamma.md").exists()
    assert conn.execute("SELECT status FROM suggestions").fetchall() == [("pending",)]


def test_audit_chunks_stay_within_the_limit_with_usage_annotations(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    root.mkdir()
    for i in range(40):
        (root / f"project_{i:02d}.md").write_text(f"Topic {i:02d} facts. " + "x" * 120 + "\n", encoding="utf-8")
    files = C.memory_files(root, constitution_path=None)
    widest = "last recalled: 2026-10-20; recalls: 999999"
    limit = 1_000
    chunks = C.audit_chunks(files, limit)
    assert sorted(entry.path.name for chunk in chunks for entry in chunk) == sorted(e.path.name for e in files)
    for chunk in chunks:
        rendered = "\n---\n".join(C.render_memory_file(entry.path.name, entry.body, annotation=widest)
                                  for entry in chunk)
        assert len(chunk) == 1 or len(rendered) <= limit
