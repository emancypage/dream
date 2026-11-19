"""`dream delegate memory`: suggestion expiry and compaction of long topic files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

import compact
import dream
from config import load_config
from model_types import ProviderError

HEAD = "---\nname: topic\n---\n"


def _body(chars, head=HEAD):
    return head + "x" * (chars - len(head))


def _setup(tmp_path, *, enabled=True, mode="auto-apply"):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "MEMORY.md").write_text("- [Topic](topic.md)\n", encoding="utf-8")
    overrides = {
        "storage": {"db_path": str(tmp_path / "dream.db"), "memory_root": str(root)},
        "review": {"mode": mode},
        "delegate": {"enabled": enabled},
        "stages": {"delegate": {"model": "fake-model"}},
    }
    cfg = load_config(tmp_path / "absent", overrides)
    conn = dream.open_db(tmp_path / "dream.db")
    args = argparse.Namespace(db=tmp_path / "dream.db", memory=str(root), model=None, config=cfg)
    return root, conn, args


def _suggest(conn, target, kind="patch", body="replacement", find="xxxx", age_days=0):
    cur = conn.execute(
        "INSERT INTO suggestions(kind, target_path, body, rationale, status, find_text, created_at) "
        "VALUES (?, ?, ?, 'why', 'pending', ?, datetime('now', ?))",
        (kind, target, body, find if kind == "patch" else None, f"-{age_days} days"),
    )
    conn.commit()
    return cur.lastrowid


def _status(conn, sug_id):
    return conn.execute("SELECT status FROM suggestions WHERE id=?", (sug_id,)).fetchone()[0]


def _fake(monkeypatch, answer):
    prompts = []

    def generate(stage, prompt, schema, **kwargs):
        prompts.append(prompt)
        return SimpleNamespace(output=answer(prompt), provider="fake", model="fake-model")

    monkeypatch.setattr(compact.backend, "generate", generate)
    return prompts


def _result(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_disabled_delegate_makes_no_call(tmp_path, monkeypatch):
    root, conn, args = _setup(tmp_path, enabled=False)
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    old = _suggest(conn, "topic.md", age_days=30)
    prompts = _fake(monkeypatch, lambda prompt: {})
    assert dream.cmd_delegate(args) == 0
    assert prompts == []
    assert _status(conn, old) == "pending"


def test_suggest_only_refuses(tmp_path, monkeypatch):
    root, conn, args = _setup(tmp_path, mode="suggest-only")
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    prompts = _fake(monkeypatch, lambda prompt: {})
    assert dream.cmd_delegate(args) == 1
    assert prompts == []
    assert (root / "topic.md").read_text(encoding="utf-8") == _body(45_000)


def test_compaction_rewrites_file_and_closes_its_suggestions(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    p1 = _suggest(conn, "topic.md", body="PATCH-ONE")
    p2 = _suggest(conn, "topic.md", body="PATCH-TWO")
    u1 = _suggest(conn, "topic.md", kind="update", body="FULL-BODY-U1")
    prompts = _fake(monkeypatch, lambda prompt: {"body": _body(20_000), "dropped_summary": "old history",
                                                 "folded_ids": [p1]})
    assert dream.cmd_delegate(args) == 0
    assert (root / "topic.md").read_text(encoding="utf-8") == _body(20_000)
    backups = list((tmp_path / "memory-backups").iterdir())
    assert len(backups) == 1 and (backups[0] / "topic.md").read_text(encoding="utf-8") == _body(45_000)
    assert (_status(conn, p1), _status(conn, p2), _status(conn, u1)) == ("accepted", "rejected", "rejected")
    rows = conn.execute("SELECT status, consolidation_run_id FROM suggestions WHERE id > ?", (u1,)).fetchall()
    assert len(rows) == 1 and rows[0][0] == "accepted" and rows[0][1].startswith("compact-")
    assert f'"id": {p1}' in prompts[0] and f'"id": {p2}' in prompts[0]
    assert "FULL-BODY-U1" not in prompts[0]
    assert _result(capsys)["compacted"][0] == {"file": "topic.md", "old_chars": 45_000, "new_chars": 20_000,
                                               "folded": [p1], "dropped": [p2, u1]}


def test_too_long_result_changes_nothing(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    p1 = _suggest(conn, "topic.md")
    _fake(monkeypatch, lambda prompt: {"body": _body(41_000), "dropped_summary": "", "folded_ids": []})
    assert dream.cmd_delegate(args) == 1
    assert (root / "topic.md").read_text(encoding="utf-8") == _body(45_000)
    assert _status(conn, p1) == "pending"
    assert not (tmp_path / "memory-backups").exists()
    assert _result(capsys)["failures"][0]["error"] == "body longer than compact_over_chars"


def test_file_changed_during_the_run_is_not_overwritten(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    topic = root / "topic.md"
    topic.write_text(_body(45_000), encoding="utf-8")
    p1 = _suggest(conn, "topic.md")

    def answer(prompt):
        with topic.open("a", encoding="utf-8") as handle:
            handle.write("\nowner edit\n")
        return {"body": _body(20_000), "dropped_summary": "", "folded_ids": [p1]}

    _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert topic.read_text(encoding="utf-8").endswith("\nowner edit\n")
    assert _status(conn, p1) == "pending"
    assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 1
    assert _result(capsys)["failures"][0]["error"] == "apply refused"


def test_provider_failure_does_not_block_the_next_file(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "big.md").write_text(_body(50_000), encoding="utf-8")
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")

    def answer(prompt):
        if "# File `big.md`" in prompt:
            raise ProviderError("x", code="exit_1:status_400:model_not_found")
        return {"body": _body(20_000), "dropped_summary": "", "folded_ids": []}

    _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert (root / "topic.md").read_text(encoding="utf-8") == _body(20_000)
    assert (root / "big.md").read_text(encoding="utf-8") == _body(50_000)
    assert _result(capsys)["failures"] == [{"file": "big.md", "error": "exit_1:status_400:model_not_found"}]


def test_only_old_suggestions_expire(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "topic.md").write_text(_body(1_000), encoding="utf-8")
    old = _suggest(conn, "topic.md", age_days=15)
    recent = _suggest(conn, "topic.md", age_days=13)
    _fake(monkeypatch, lambda prompt: {})
    assert dream.cmd_delegate(args) == 0
    assert (_status(conn, old), _status(conn, recent)) == ("rejected", "pending")
    assert _result(capsys)["expired"] == [old]


def test_index_file_is_never_compacted(tmp_path, monkeypatch):
    root, conn, args = _setup(tmp_path)
    (root / "MEMORY.md").write_text(_body(50_000, head=""), encoding="utf-8")
    prompts = _fake(monkeypatch, lambda prompt: {})
    assert dream.cmd_delegate(args) == 0
    assert prompts == []


def test_renamed_frontmatter_is_refused(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    _fake(monkeypatch, lambda prompt: {"body": _body(20_000, head="---\nname: other\n---\n"),
                                       "dropped_summary": "", "folded_ids": []})
    assert dream.cmd_delegate(args) == 1
    assert (root / "topic.md").read_text(encoding="utf-8") == _body(45_000)
    assert _result(capsys)["failures"][0]["error"] == "frontmatter changed"


ROOT = Path(__file__).parent


def test_nightly_unit_runs_compaction_between_distill_and_consolidate():
    lines = [l for l in (ROOT / "systemd" / "dream.service").read_text().splitlines() if l.startswith("ExecStart=")]
    stages = [l.split()[2] for l in lines]
    assert stages.index("distill") < stages.index("delegate_memory") < stages.index("consolidate")


def _preflight(tmp_path, monkeypatch, capsys, overrides):
    monkeypatch.setattr(dream, "preflight", lambda config: [])
    monkeypatch.setattr(dream, "recall_preflight", lambda config, db: [])
    cfg = load_config(tmp_path / "absent", overrides)
    code = dream.cmd_preflight(argparse.Namespace(config=cfg, db=tmp_path / "dream.db"))
    return code, capsys.readouterr().out.splitlines()


def test_preflight_reports_the_delegate_off(tmp_path, monkeypatch, capsys):
    code, lines = _preflight(tmp_path, monkeypatch, capsys, {})
    assert code == 0 and "ok  delegate: off" in lines


def test_preflight_reports_memory_compaction(tmp_path, monkeypatch, capsys):
    code, lines = _preflight(tmp_path, monkeypatch, capsys, {
        "review": {"mode": "auto-apply"}, "delegate": {"enabled": True}, "stages": {"delegate": {"model": "m"}}})
    assert code == 0 and "ok  delegate: memory compaction over 40000 characters" in lines


def test_preflight_fails_the_delegate_with_suggest_only(tmp_path, monkeypatch, capsys):
    code, lines = _preflight(tmp_path, monkeypatch, capsys, {
        "review": {"mode": "suggest-only"}, "delegate": {"enabled": True}, "stages": {"delegate": {"model": "m"}}})
    assert code == 1 and "FAIL  delegate: requires review.mode = auto-apply" in lines


@pytest.mark.parametrize("field", ["body", "rationale", "find_text", "source_sessions"])
def test_retention_excluded_suggestion_fields_are_not_sent_or_decided(tmp_path, monkeypatch, capsys, field):
    root, conn, args = _setup(tmp_path)
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    (root / ".dream-retention.json").write_text(json.dumps({"text_markers": ["PRIVATE-MARKER"], "project_slugs": ["excluded"]}))
    sug_id = _suggest(conn, "topic.md")
    value = "PRIVATE-MARKER"
    if field == "source_sessions":
        conn.execute("INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime) VALUES ('excluded-session', 'excluded', '/synthetic/excluded.jsonl', 0)")
        value = "excluded-session"
    conn.execute(f"UPDATE suggestions SET {field}=? WHERE id=?", (value, sug_id))
    conn.commit()
    prompts = _fake(monkeypatch, lambda prompt: {"body": _body(20_000), "dropped_summary": "", "folded_ids": []})
    assert dream.cmd_delegate(args) == 0
    assert f'"id": {sug_id}' not in prompts[0]
    assert _status(conn, sug_id) == "pending"
    assert _result(capsys)["compacted"][0]["dropped"] == []


@pytest.mark.parametrize("change", ["reject", "find", "body", "new"])
def test_changed_pending_group_aborts_compaction(tmp_path, monkeypatch, capsys, change):
    root, conn, args = _setup(tmp_path)
    topic = root / "topic.md"
    topic.write_text(_body(45_000), encoding="utf-8")
    sug_id = _suggest(conn, "topic.md")

    def answer(prompt):
        writer = dream.open_db(args.db)
        if change == "new":
            _suggest(writer, "topic.md")
        else:
            field, value = {"reject": ("status", "rejected"), "find": ("find_text", "changed"),
                            "body": ("body", "changed")}[change]
            writer.execute(f"UPDATE suggestions SET {field}=? WHERE id=?", (value, sug_id))
            writer.commit()
        writer.close()
        return {"body": _body(20_000), "dropped_summary": "", "folded_ids": [sug_id]}

    _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert topic.read_text() == _body(45_000)
    assert _status(conn, sug_id) == ("rejected" if change == "reject" else "pending")
    assert _result(capsys)["failures"] == [{"file": "topic.md", "error": "pending suggestions changed"}]
    assert not (tmp_path / "memory-backups").exists()


@pytest.mark.parametrize("change", ["policy", "body", "dropped_summary"])
def test_retention_is_rechecked_after_generation(tmp_path, monkeypatch, capsys, change):
    root, conn, args = _setup(tmp_path)
    topic = root / "topic.md"
    topic.write_text(_body(45_000), encoding="utf-8")
    policy = root / ".dream-retention.json"
    policy.write_text(json.dumps({"text_markers": ["PRIVATE-MARKER"]}))
    sug_id = _suggest(conn, "topic.md")

    def answer(prompt):
        output = {"body": _body(20_000), "dropped_summary": "", "folded_ids": [sug_id]}
        if change == "policy":
            policy.write_text(json.dumps({"memory_paths": ["topic.md"]}))
        else:
            output[change] += "PRIVATE-MARKER"
        return output

    _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert topic.read_text() == _body(45_000)
    assert _status(conn, sug_id) == "pending"
    assert _result(capsys)["compacted"] == []
    assert not (tmp_path / "memory-backups").exists()


def test_policy_change_between_files_prevents_next_model_call(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "big.md").write_text(_body(50_000), encoding="utf-8")
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")

    def answer(prompt):
        (root / ".dream-retention.json").write_text(json.dumps({"memory_paths": ["topic.md"]}))
        return {"body": _body(20_000), "dropped_summary": "", "folded_ids": []}

    prompts = _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert len(prompts) == 1
    assert (root / "topic.md").read_text() == _body(45_000)
    assert _result(capsys)["compacted"] == []


def test_oversized_files_are_reported_without_starving_eligible_topics(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    for name, chars in [("huge.md", 200_001), ("big.md", 200_002), ("topic.md", 45_000), ("other.md", 44_000), ("last.md", 43_000)]:
        (root / name).write_text(_body(chars), encoding="utf-8")
    prompts = _fake(monkeypatch, lambda prompt: {"body": _body(20_000), "dropped_summary": "", "folded_ids": []})
    assert dream.cmd_delegate(args) == 1
    result = _result(capsys)
    assert len(prompts) == 2
    assert {item["file"] for item in result["compacted"]} == {"topic.md", "other.md"}
    assert {item["file"] for item in result["failures"]} == {"huge.md", "big.md"}


def test_recall_finalization_failure_keeps_compacted_list_and_fails_stage(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")
    _fake(monkeypatch, lambda prompt: {"body": _body(20_000), "dropped_summary": "", "folded_ids": []})

    monkeypatch.setattr(dream, "_sync_recall_documents", lambda *args, **kwargs: False)
    assert dream.cmd_delegate(args) == 1
    result = _result(capsys)
    assert result["compacted"][0]["file"] == "topic.md"
    assert (root / "topic.md").read_text() == _body(20_000)
    assert result["failures"][0]["error"].startswith("recall synchronization failed")
    assert "private detail" not in json.dumps(result)


@pytest.mark.parametrize("change", ["shrink", "delete"])
def test_later_topic_changed_by_earlier_call_is_not_sent(tmp_path, monkeypatch, capsys, change):
    root, conn, args = _setup(tmp_path)
    (root / "big.md").write_text(_body(50_000), encoding="utf-8")
    topic = root / "topic.md"
    topic.write_text(_body(45_000), encoding="utf-8")

    def answer(prompt):
        if change == "shrink":
            topic.write_text("owner edit", encoding="utf-8")
        else:
            topic.unlink()
        return {"body": _body(20_000), "dropped_summary": "", "folded_ids": []}

    prompts = _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert len(prompts) == 1
    result = _result(capsys)
    assert result["failures"] == [{"file": "topic.md", "error": "apply refused"}]
    assert (topic.read_text() if topic.exists() else None) == ("owner edit" if change == "shrink" else None)


def test_topic_deleted_during_generation_does_not_block_next_file(tmp_path, monkeypatch, capsys):
    root, conn, args = _setup(tmp_path)
    big = root / "big.md"
    big.write_text(_body(50_000), encoding="utf-8")
    (root / "topic.md").write_text(_body(45_000), encoding="utf-8")

    def answer(prompt):
        if "# File `big.md`" in prompt:
            big.unlink()
        return {"body": _body(20_000), "dropped_summary": "", "folded_ids": []}

    prompts = _fake(monkeypatch, answer)
    assert dream.cmd_delegate(args) == 1
    assert len(prompts) == 2
    result = _result(capsys)
    assert result["compacted"][0]["file"] == "topic.md"
    assert result["failures"] == [{"file": "big.md", "error": "apply refused"}]
    assert not big.exists()


def test_expiry_commits_each_status_before_preview_deletion_error(tmp_path, monkeypatch):
    root, conn, args = _setup(tmp_path)
    sug_dir = root / ".suggestions"
    sug_dir.mkdir()
    ids = [_suggest(conn, "topic.md", age_days=15) for _ in range(2)]
    previews = [sug_dir / f"{sug_id}.md" for sug_id in ids]
    for sug_id, preview in zip(ids, previews):
        preview.write_text("synthetic preview", encoding="utf-8")
        conn.execute("UPDATE suggestions SET sug_file=? WHERE id=?", (preview.name, sug_id))
    conn.commit()
    unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if path == previews[1]:
            raise OSError("synthetic deletion error")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(OSError, match="synthetic deletion error"):
        dream.expire_suggestions(conn, root, 14)
    reader = dream.open_db(args.db)
    try:
        assert [_status(reader, sug_id) for sug_id in ids] == ["rejected", "rejected"]
    finally:
        reader.close()
    assert not conn.in_transaction
    assert not previews[0].exists()
    assert previews[1].exists()


def test_expiry_skips_suggestion_reviewed_after_selection(tmp_path, monkeypatch):
    root, conn, args = _setup(tmp_path)
    first, second = [_suggest(conn, "topic.md", age_days=15) for _ in range(2)]
    removed = []

    def drop_preview(sug_dir, sug_file):
        removed.append(sug_file)
        writer = dream.open_db(args.db)
        try:
            writer.execute("UPDATE suggestions SET status='accepted' WHERE id=?", (second,))
            writer.commit()
        finally:
            writer.close()

    monkeypatch.setattr(dream, "_drop_suggestion_preview", drop_preview)
    assert dream.expire_suggestions(conn, root, 14) == [first]
    assert _status(conn, first) == "rejected"
    assert _status(conn, second) == "accepted"
    assert removed == [None]
