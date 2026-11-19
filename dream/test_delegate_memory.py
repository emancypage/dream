"""`dream delegate memory`: suggestion expiry and compaction of long topic files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert _result(capsys)["failures"][0]["error"] == "frontmatter name changed"
