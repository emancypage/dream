"""The weekly staleness audit: chunking, allowed kinds, the grace gate and its units."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import consolidate as C
import dream
from config import load_config


ROOT = Path(__file__).parent
REPO = ROOT.parent
SCHEMA = (ROOT / "schema.sql").read_text(encoding="utf-8")
FILE_HEADER = re.compile(r"^### FILE: (.+)$", re.MULTILINE)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    dream._migrate(conn)
    return conn


def _age(path: Path, days: int) -> None:
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).timestamp()
    os.utime(path, (ts, ts))


def _store(tmp_path, count=60):
    root = tmp_path / "memory"
    root.mkdir()
    index = []
    for i in range(count):
        name = f"project_topic_{i:02d}.md"
        (root / name).write_text(
            f"---\nname: topic-{i:02d}\nmetadata:\n  type: project\n---\n\nTopic {i:02d} facts. Marker BODY-{i:02d}.\n",
            encoding="utf-8",
        )
        index.append(f"- [Topic {i:02d}]({name}) — hook {i:02d}")
    (root / "MEMORY.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (root / ".hidden.md").write_text("never audited", encoding="utf-8")
    return root


def _config(tmp_path, root, mode="subset"):
    return load_config(tmp_path / "missing-config.toml", overrides={
        "storage": {"db_path": str(tmp_path / "dream.db"), "memory_root": str(root)},
        "consolidate": {"mode": mode},
    })


def _fake(monkeypatch, outputs=None):
    calls = []

    def fake_generate(stage, prompt, schema, **kwargs):
        calls.append({"stage": stage, "prompt": prompt, "schema": schema})
        output = outputs(len(calls), prompt) if outputs else {"suggestions": []}
        return SimpleNamespace(output=output, provider="fake", model="fake-model")

    monkeypatch.setattr(C, "generate", fake_generate)
    return calls


def _suggestion(kind, target, body="", rationale="DEPRECATED: closed", find=""):
    return {"kind": kind, "target_path": target, "find": find, "body": body,
            "rationale": rationale, "source_sessions": []}


def test_audit_chunks_cover_every_topic_file_exactly_once(tmp_path, monkeypatch):
    root = _store(tmp_path)
    monkeypatch.setattr(C, "MAX_CURRENT_MEMORY_CHARS", 4 * 1_000)
    calls = _fake(monkeypatch)
    C.run_audit(_db(), root, config=_config(tmp_path, root))
    assert len(calls) > 5
    seen = [name for call in calls for name in FILE_HEADER.findall(call["prompt"])]
    assert sorted(seen) == sorted(f"project_topic_{i:02d}.md" for i in range(60))
    assert len(seen) == len(set(seen))
    for number, call in enumerate(calls, 1):
        assert call["stage"] == "audit"
        assert call["schema"] is C.CONSOLIDATE_PATCH_SCHEMA
        assert f"chunk {number} of {len(calls)}" in call["prompt"]
        assert "- [Topic 59](project_topic_59.md) — hook 59" in call["prompt"]
        assert "### FILE: MEMORY.md" not in call["prompt"]
        assert "never audited" not in call["prompt"]
        assert "{{" not in call["prompt"]


def test_audit_chunks_hold_whole_files_within_the_limit(tmp_path):
    root = _store(tmp_path)
    (root / "project_huge.md").write_text("h" * 5_000, encoding="utf-8")
    files = C.memory_files(root, constitution_path=None)
    chunks = C.audit_chunks(files, 1_000)
    names = [entry.path.name for chunk in chunks for entry in chunk]
    assert sorted(names) == sorted(entry.path.name for entry in files)
    for chunk in chunks:
        rendered = "\n---\n".join(C.render_memory_file(entry.path.name, entry.body) for entry in chunk)
        if len(chunk) > 1:
            assert len(rendered) <= 1_000
        else:
            assert chunk[0].body == (root / chunk[0].path.name).read_text(encoding="utf-8")
    [huge_chunk] = [chunk for chunk in chunks if any(entry.path.name == "project_huge.md" for entry in chunk)]
    assert len(huge_chunk) == 1


def test_default_audit_chunk_limit_is_a_quarter_of_the_memory_cap(tmp_path, monkeypatch):
    root = _store(tmp_path, count=3)
    seen = []
    real = C.audit_chunks
    monkeypatch.setattr(C, "audit_chunks", lambda files, limit: seen.append(limit) or real(files, limit))
    _fake(monkeypatch)
    C.run_audit(_db(), root, config=_config(tmp_path, root))
    assert seen == [C.MAX_CURRENT_MEMORY_CHARS // 4] == [200_000]


def test_audit_emits_only_remove_and_patch(tmp_path, monkeypatch, capsys):
    root = _store(tmp_path, count=5)
    for i in range(5):
        _age(root / f"project_topic_{i:02d}.md", 40)
    _fake(monkeypatch, lambda n, prompt: {"suggestions": [
        _suggestion("new", "project_new.md", "new body"),
        _suggestion("update", "project_topic_00.md", "rewritten"),
        _suggestion("index", "MEMORY.md", "- [X](x.md) — x\n"),
        _suggestion("remove", "project_topic_01.md"),
        _suggestion("patch", "project_topic_02.md", "Topic 02 facts, revised.", find="Topic 02 facts."),
    ]})
    conn = _db()
    suggestions = C.run_audit(conn, root, config=_config(tmp_path, root))
    assert [(s.kind, s.target_path) for s in suggestions] == [
        ("remove", "project_topic_01.md"), ("patch", "project_topic_02.md")]
    assert [row for row in conn.execute("SELECT kind, find_text FROM suggestions ORDER BY id")] == [
        ("remove", None), ("patch", "Topic 02 facts.")]
    out = capsys.readouterr().out
    for dropped in ("new project_new.md", "update project_topic_00.md", "index MEMORY.md"):
        assert f"dropped {dropped}" in out


@pytest.mark.parametrize("bad", [
    _suggestion("remove", "MEMORY.md"),
    _suggestion("patch", "MEMORY.md", "x", find="- [Topic 00]"),
    _suggestion("patch", "project_topic_01.md", "x", find=""),
    _suggestion("remove", "project_zz_later_chunk.md"),
    _suggestion("remove", ".dream-retention.json"),
])
def test_audit_drops_suggestions_for_files_it_was_not_shown(tmp_path, monkeypatch, bad):
    root = _store(tmp_path, count=3)
    (root / "project_zz_later_chunk.md").write_text("x" * 400, encoding="utf-8")
    (root / ".dream-retention.json").write_text("{}", encoding="utf-8")
    for path in root.glob("*.md"):
        _age(path, 40)
    monkeypatch.setattr(C, "MAX_CURRENT_MEMORY_CHARS", 4 * 300)
    _fake(monkeypatch, lambda n, prompt: {"suggestions": [bad]} if n == 1 else {"suggestions": []})
    conn = _db()
    assert C.run_audit(conn, root, config=_config(tmp_path, root)) == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0


def test_audit_drops_a_patch_that_would_empty_a_file(tmp_path, monkeypatch, capsys):
    root = _store(tmp_path, count=2)
    body = "Topic 00 facts. Marker BODY-00.\n"
    _fake(monkeypatch, lambda n, prompt: {"suggestions": [
        _suggestion("patch", "project_topic_00.md", "", "delete obsolete file", find=body)]})
    conn = _db()
    assert C.run_audit(conn, root, config=_config(tmp_path, root)) == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0
    out = capsys.readouterr().out
    assert "dropped patch project_topic_00.md" in out and "would empty the file" in out


def test_audit_never_persists_retention_excluded_find_text(tmp_path, monkeypatch):
    root = _store(tmp_path, count=2)
    (root / ".dream-retention.json").write_text('{"text_markers": ["SECRET-Y"]}', encoding="utf-8")
    _fake(monkeypatch, lambda n, prompt: {"suggestions": [
        _suggestion("patch", "project_topic_00.md", "Topic 00 facts.", find="Topic 00 facts. SECRET-Y")]})
    conn = _db()
    assert C.run_audit(conn, root, config=_config(tmp_path, root)) == []
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0
    assert not list(root.glob(".suggestions/*.md"))


def test_audit_base_sha_is_the_hash_of_the_bytes_the_model_saw(tmp_path, monkeypatch, capsys):
    root = _store(tmp_path, count=2)
    real_files = C.memory_files

    def read_then_edit(*args, **kwargs):
        files = real_files(*args, **kwargs)
        (root / "project_topic_00.md").write_text("---\nname: topic-00\n---\n\nEdited by hand.\n", encoding="utf-8")
        return files

    monkeypatch.setattr(C, "memory_files", read_then_edit)
    calls = _fake(monkeypatch, lambda n, prompt: {"suggestions": [
        _suggestion("patch", "project_topic_00.md", "Topic 00 facts, revised.", find="Topic 00 facts.")]})
    db_path = tmp_path / "dream.db"
    conn = dream.open_db(db_path)
    [s] = C.run_audit(conn, root, config=_config(tmp_path, root))
    conn.close()
    assert "BODY-00" in calls[0]["prompt"]
    args = argparse.Namespace(db=db_path, memory=str(root), suggestions_cmd="accept", id=1,
                              config=_config(tmp_path, root))
    assert dream.cmd_suggestions(args) == 3
    assert "conflict #1" in capsys.readouterr().out
    assert "Edited by hand." in (root / "project_topic_00.md").read_text(encoding="utf-8")


def test_audit_removals_keep_the_21_day_grace(tmp_path, monkeypatch):
    root = _store(tmp_path, count=2)
    _age(root / "project_topic_00.md", C.REMOVE_GRACE_DAYS + 5)
    _fake(monkeypatch, lambda n, prompt: {"suggestions": [
        _suggestion("remove", "project_topic_00.md"), _suggestion("remove", "project_topic_01.md")]})
    suggestions = C.run_audit(_db(), root, config=_config(tmp_path, root))
    assert [s.target_path for s in suggestions] == ["project_topic_00.md"]


def test_audit_records_a_run_and_consumes_no_distillations(tmp_path, monkeypatch):
    root = _store(tmp_path, count=2)
    conn = _db()
    conn.execute(
        "INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime) VALUES ('s-1', 'p', '/p', 1)")
    conn.execute("INSERT INTO distilled(session_id, distillation_key, notes_json) VALUES ('s-1', 'k-1', '{}')")
    conn.commit()
    _fake(monkeypatch)
    C.run_audit(conn, root, config=_config(tmp_path, root))
    assert conn.execute("SELECT provider, model FROM consolidation_runs").fetchall() == [("fake", "fake-model")]
    assert conn.execute("SELECT COUNT(*) FROM consolidated_distillations").fetchone()[0] == 0
    assert [row[0] for row in C.collect_distilled_batch(conn, None)] == ["s-1"]


def test_audit_provider_failure_persists_nothing(tmp_path, monkeypatch):
    root = _store(tmp_path, count=10)
    for path in root.glob("*.md"):
        _age(path, 40)
    monkeypatch.setattr(C, "MAX_CURRENT_MEMORY_CHARS", 4 * 300)

    def flaky(stage, prompt, schema, **kwargs):
        if "chunk 1 of" in prompt:
            return SimpleNamespace(output={"suggestions": [_suggestion("remove", FILE_HEADER.findall(prompt)[0])]},
                                   provider="fake", model="fake")
        raise RuntimeError("provider failed")

    monkeypatch.setattr(C, "generate", flaky)
    conn = _db()
    with pytest.raises(RuntimeError, match="provider failed"):
        C.run_audit(conn, root, config=_config(tmp_path, root))
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM consolidation_runs").fetchone()[0] == 0
    assert not list(root.glob(".suggestions/*.md"))


def test_audit_of_an_empty_store_makes_no_provider_call(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "MEMORY.md").write_text("", encoding="utf-8")
    calls = _fake(monkeypatch)
    assert C.run_audit(_db(), root, config=_config(tmp_path, root)) == []
    assert calls == []


def test_audit_prompt_keeps_the_age_rule_and_has_no_session_batch(tmp_path, monkeypatch):
    root = _store(tmp_path, count=2)
    calls = _fake(monkeypatch)
    C.run_audit(_db(), root, config=_config(tmp_path, root))
    prompt = calls[0]["prompt"]
    assert "Do NOT use file age" in prompt
    assert "ODRZUCONE" in prompt
    assert "`remove`" in prompt and "`patch`" in prompt
    assert "distilled" not in prompt.lower()


def test_every_memory_file_in_audit_and_subset_prompts_renders_through_one_function(tmp_path, monkeypatch):
    root = _store(tmp_path, count=3)
    real = C.render_memory_file
    monkeypatch.setattr(C, "render_memory_file",
                        lambda name, body, **kw: real(name, body, annotation=f"annotated: {name}"))
    calls = _fake(monkeypatch)
    C.run_audit(_db(), root, config=_config(tmp_path, root))
    for i in range(3):
        assert f"### FILE: project_topic_{i:02d}.md\nannotated: project_topic_{i:02d}.md\n" in calls[0]["prompt"]
    assert C.read_current_memory(root, constitution_path=None).count("annotated: ") == 4


# ------------------------------------------------------------ dream audit


def _audit_args(tmp_path, root, config, force=False):
    return argparse.Namespace(db=tmp_path / "dream.db", memory=str(root), model=None, force=force, config=config)


def test_dream_audit_skips_in_full_mode_unless_forced(tmp_path, monkeypatch, capsys):
    root = _store(tmp_path, count=2)
    calls = _fake(monkeypatch)
    config = _config(tmp_path, root, mode="full")
    assert dream.cmd_audit(_audit_args(tmp_path, root, config)) == 0
    assert calls == []
    assert "consolidate.mode is full" in capsys.readouterr().out
    assert dream.cmd_audit(_audit_args(tmp_path, root, config, force=True)) == 0
    assert len(calls) == 1


def test_dream_audit_runs_in_subset_mode_through_the_cli(tmp_path, monkeypatch, capsys):
    root = _store(tmp_path, count=2)
    _age(root / "project_topic_00.md", 40)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "dream.db"}"\nmemory_root = "{root}"\n\n[consolidate]\nmode = "subset"\n',
        encoding="utf-8",
    )
    calls = _fake(monkeypatch, lambda n, prompt: {"suggestions": [_suggestion("remove", "project_topic_00.md")]})
    assert dream.main(["--config", str(config_path), "audit"]) == 0
    assert len(calls) == 1
    assert "project_topic_00.md" in capsys.readouterr().out
    conn = sqlite3.connect(tmp_path / "dream.db")
    try:
        assert conn.execute("SELECT kind, target_path, status FROM suggestions").fetchall() == [
            ("remove", "project_topic_00.md", "pending")]
    finally:
        conn.close()
    assert (root / "project_topic_00.md").exists()


# ------------------------------------------------------------ configuration and units


def test_default_config_routes_the_audit_stage_like_consolidate():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    assert config.stage("audit") == {"provider": "codex", "reasoning_effort": "high", "timeout_seconds": 1800}


def _exec_lines(text):
    return [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("ExecStart")]


def test_audit_service_runs_dream_audit_directly_without_provider_commands():
    service = (ROOT / "systemd" / "dream-audit.service").read_text(encoding="utf-8")
    assert _exec_lines(service) == ["%h/.local/bin/dream audit"]
    assert "Type=oneshot" in service
    lowered = service.lower()
    for forbidden in ("codex", "claude", "opencode", "dream-stage", "dream-metrics", "--force"):
        assert forbidden not in lowered


def test_audit_timer_is_weekly_and_persistent():
    timer = (ROOT / "systemd" / "dream-audit.timer").read_text(encoding="utf-8")
    calendar = [line.split("=", 1)[1] for line in timer.splitlines() if line.startswith("OnCalendar=")]
    assert len(calendar) == 1
    assert calendar[0].split()[0] in {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"} or calendar[0] == "weekly"
    assert "Persistent=true" in timer
    assert "WantedBy=timers.target" in timer
    lowered = timer.lower()
    for forbidden in ("codex", "claude", "opencode", "execstart"):
        assert forbidden not in lowered


def test_install_copies_the_audit_units_without_enabling_anything():
    install = (REPO / "install.sh").read_text(encoding="utf-8")
    for unit in ("dream-audit.service", "dream-audit.timer"):
        assert f'"$REPO_DIR/dream/systemd/{unit}" "$HOME/.config/systemd/user/{unit}"' in install
    assert "systemctl" not in install.replace("run systemctl --user daemon-reload separately", "")


def test_nightly_service_is_unchanged_by_the_audit():
    service = (ROOT / "systemd" / "dream.service").read_text(encoding="utf-8")
    assert "dream audit" not in service
    assert "%h/.local/bin/dream consolidate" in service
