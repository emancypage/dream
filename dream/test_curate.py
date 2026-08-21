import argparse
import hashlib
import sqlite3
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import dream  # noqa: E402
from config import load_config  # noqa: E402
from dream import open_db  # noqa: E402

from curate import (  # noqa: E402
    CURATION_SCHEMA,
    CurationDecision,
    MAX_CURATION_FIELD_CHARS,
    build_curation_prompt,
    parse_curation_output,
)
from model_types import SchemaValidationError, validate_output  # noqa: E402


def _decision(suggestion_id, decision="defer", reason="needs more evidence", **extra):
    value = {
        "suggestion_id": suggestion_id,
        "decision": decision,
        "reason": reason,
    }
    value.update(extra)
    return value


def test_parse_curation_output_returns_typed_decisions():
    result = parse_curation_output(
        {"decisions": [_decision(7, "merge", "keeps newer facts", body="merged")]},
        {7},
    )

    assert result == {
        7: CurationDecision(
            suggestion_id=7,
            decision="merge",
            reason="keeps newer facts",
            body="merged",
        )
    }


@pytest.mark.parametrize(
    ("output", "expected_ids", "message"),
    [
        ({"decisions": [_decision(7), _decision(7)]}, {7}, "duplicate"),
        ({"decisions": [_decision(8)]}, {7}, "suggestion IDs"),
        ({"decisions": [_decision(7), _decision(8)]}, {7}, "suggestion IDs"),
    ],
)
def test_parse_curation_output_rejects_duplicate_missing_and_unknown_ids(
    output, expected_ids, message
):
    with pytest.raises(ValueError, match=message):
        parse_curation_output(output, expected_ids)


def test_parse_curation_output_rejects_body_for_non_merge_decision():
    with pytest.raises(ValueError, match="body.*accept"):
        parse_curation_output(
            {"decisions": [_decision(7, "accept", "matches current facts", body="wrong")]},
            {7},
        )


def test_curation_schema_rejects_extra_keys_at_both_object_levels():
    valid = {"decisions": [_decision(7)]}

    with pytest.raises(SchemaValidationError, match="unexpected keys: extra"):
        validate_output({**valid, "extra": True}, CURATION_SCHEMA)
    with pytest.raises(SchemaValidationError, match="unexpected keys: extra"):
        validate_output(
            {"decisions": [{**_decision(7), "extra": True}]}, CURATION_SCHEMA
        )


def test_curation_schema_requires_non_empty_reason():
    with pytest.raises(ValueError, match="reason.*non-empty"):
        parse_curation_output(
            {"decisions": [_decision(7, reason="   ")]},
            {7},
        )


def test_build_curation_prompt_contains_bounded_review_context_and_safety_rules():
    prompt = build_curation_prompt(
        [
            {
                "id": 7,
                "kind": "update",
                "target_path": "topics/python.md",
                "body": "proposal body",
                "rationale": "new evidence",
                "source_sessions": ["session-a", "session-b"],
                "current_body": "live body",
                "current_sha256": "live-sha",
                "base_sha256": "base-sha",
                "target_existed": True,
                "conflict": True,
            }
        ],
        Path("/memory/root"),
    )

    for expected in (
        "7",
        "update",
        "topics/python.md",
        "proposal body",
        "new evidence",
        "session-a",
        "session-b",
        "live body",
        "live-sha",
        "base-sha",
        '"target_existed": true',
        '"conflict": true',
    ):
        assert expected in prompt
    assert "never invent" in prompt
    assert "MEMORY.md" in prompt
    assert "only intended index lines" in prompt
    assert "complete replacement body" in prompt
    assert "accept, reject, merge, defer" in prompt
    assert "concise" in prompt


def test_build_curation_prompt_rejects_field_overrun_before_serializing():
    with pytest.raises(ValueError, match="suggestion ID 7.*proposal_body"):
        build_curation_prompt(
            [{"id": 7, "body": "x" * (MAX_CURATION_FIELD_CHARS + 1)}],
            Path("/memory/root"),
        )


def test_build_curation_prompt_rejects_complete_prompt_overrun():
    rows = [
        {"id": suggestion_id, "body": "x" * (MAX_CURATION_FIELD_CHARS - 1)}
        for suggestion_id in range(1, 6)
    ]

    with pytest.raises(ValueError, match="prompt limit"):
        build_curation_prompt(rows, Path("/memory/root"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _insert_pending(conn, root, target, kind, body, *, preview=True):
    path = root / target
    existed = path.exists()
    preview_name = f"preview-{Path(target).stem}.md" if preview else None
    cur = conn.execute(
        "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, "
        "status, sug_file, base_sha256, target_existed) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
        (kind, target, body, "fixture rationale", "session-a,session-b",
         preview_name,
         _sha(path), int(existed)),
    )
    if preview:
        sug_dir = root / ".suggestions"
        sug_dir.mkdir(exist_ok=True)
        (sug_dir / preview_name).write_text(
            body, encoding="utf-8"
        )
    return cur.lastrowid


def make_curation_fixture(tmp_path, auto_apply=True, target=None):
    root = tmp_path / "memory"
    root.mkdir()
    db_path = tmp_path / "dream.db"
    config = load_config(
        tmp_path / "missing-config.toml",
        overrides={
            "storage": {"db_path": str(db_path), "memory_root": str(root)},
            "review": {"mode": "auto-apply" if auto_apply else "suggest-only"},
        },
    )
    conn = open_db(db_path)
    if target:
        (root / target).write_text("original conflict body", encoding="utf-8")
        _insert_pending(conn, root, target, "update", "stored proposal")
    else:
        for name, body in (
            ("accept.md", "original accept body"),
            ("merge.md", "original merge body"),
            ("reject.md", "original reject body"),
            ("defer.md", "original defer body"),
        ):
            (root / name).write_text(body, encoding="utf-8")
        _insert_pending(conn, root, "accept.md", "update", "stored accept proposal")
        _insert_pending(conn, root, "merge.md", "update", "stored merge proposal")
        _insert_pending(conn, root, "reject.md", "update", "stored reject proposal")
        _insert_pending(conn, root, "defer.md", "update", "stored defer proposal")
    conn.commit()
    conn.close()
    return root, db_path, config


def run_curate(root, db_path, config, dry_run=False):
    return dream.cmd_suggestions(
        argparse.Namespace(
            db=db_path,
            memory=str(root),
            suggestions_cmd="curate-configured",
            config=config,
            dry_run=dry_run,
        )
    )


def statuses(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return dict(conn.execute("SELECT id, status FROM suggestions"))
    finally:
        conn.close()


def _fixture_snapshot(root, db_path):
    files = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    return files, db_path.read_bytes()


def test_curate_accept_merge_reject_and_defer_updates_only_expected_rows(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True)

    def fake_review(stage, prompt, schema, **kwargs):
        assert stage == "review"
        assert schema is CURATION_SCHEMA
        assert kwargs["config"] is config
        return SimpleNamespace(
            output={
                "decisions": [
                    {"suggestion_id": 1, "decision": "accept", "reason": "confirmed"},
                    {"suggestion_id": 2, "decision": "merge", "reason": "updated", "body": "model merged body"},
                    {"suggestion_id": 3, "decision": "reject", "reason": "obsolete"},
                    {"suggestion_id": 4, "decision": "defer", "reason": "needs review"},
                ]
            },
            provider="codex",
            model="gpt-5.6-luna",
        )

    monkeypatch.setattr(dream, "generate", fake_review, raising=False)
    assert run_curate(root, db_path, config) == 0
    assert (root / "accept.md").read_text(encoding="utf-8") == "stored accept proposal"
    assert (root / "merge.md").read_text(encoding="utf-8") == "model merged body"
    assert (root / "reject.md").read_text(encoding="utf-8") == "original reject body"
    assert statuses(db_path) == {1: "accepted", 2: "accepted", 3: "rejected", 4: "pending"}
    assert not (root / ".suggestions" / "preview-accept.md").exists()
    assert not (root / ".suggestions" / "preview-merge.md").exists()
    assert not (root / ".suggestions" / "preview-reject.md").exists()
    assert (root / ".suggestions" / "preview-defer.md").exists()
    assert list(tmp_path.glob("memory-backups/*"))


def test_curate_conflict_rechecks_sha_and_defers_without_overwrite(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True, target="conflict.md")

    def fake_review(stage, prompt, schema, **kwargs):
        (root / "conflict.md").write_text("new live body", encoding="utf-8")
        return SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "merge", "reason": "preserve live edit", "body": "model body"}]},
            provider="codex",
            model="gpt-5.6-luna",
        )

    monkeypatch.setattr(dream, "generate", fake_review, raising=False)
    assert run_curate(root, db_path, config) == 0
    assert (root / "conflict.md").read_text(encoding="utf-8") == "new live body"
    assert statuses(db_path)[1] == "pending"
    assert not list(tmp_path.glob("memory-backups/*"))


def test_curate_provider_failure_writes_nothing(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True)
    before = _fixture_snapshot(root, db_path)

    def raise_provider_error(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(dream, "generate", raise_provider_error, raising=False)
    assert run_curate(root, db_path, config) != 0
    assert _fixture_snapshot(root, db_path) == before
    assert not list(tmp_path.glob("memory-backups/*"))


def test_curate_dry_run_validates_and_writes_nothing(tmp_path, monkeypatch, capsys):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True)
    before = _fixture_snapshot(root, db_path)

    monkeypatch.setattr(
        dream,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            output={"decisions": [
                {"suggestion_id": 1, "decision": "accept", "reason": "confirmed"},
                {"suggestion_id": 2, "decision": "merge", "reason": "updated", "body": "merged"},
                {"suggestion_id": 3, "decision": "reject", "reason": "obsolete"},
                {"suggestion_id": 4, "decision": "defer", "reason": "later"},
            ]},
            provider="codex", model="gpt-5.6-luna",
        ),
        raising=False,
    )

    assert run_curate(root, db_path, config, dry_run=True) == 0
    output = capsys.readouterr().out
    assert "codex / gpt-5.6-luna" in output
    assert "backup:" not in output.lower()
    assert _fixture_snapshot(root, db_path) == before
    assert not list(tmp_path.glob("memory-backups/*"))


def test_curate_rejects_incomplete_response_before_any_mutation(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True)
    before = _fixture_snapshot(root, db_path)
    monkeypatch.setattr(
        dream,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "reject", "reason": "obsolete"}]},
            provider="codex", model="gpt-5.6-luna",
        ),
        raising=False,
    )

    assert run_curate(root, db_path, config) != 0
    assert _fixture_snapshot(root, db_path) == before
    assert not list(tmp_path.glob("memory-backups/*"))


def test_curate_index_merge_keeps_existing_lines_and_appends_model_lines(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "MEMORY.md").write_text("- [Existing](existing.md) — keep\n", encoding="utf-8")
    (root / "existing.md").write_text("existing", encoding="utf-8")
    (root / "new.md").write_text("new", encoding="utf-8")
    db_path = tmp_path / "dream.db"
    config = load_config(
        tmp_path / "missing-config.toml",
        overrides={
            "storage": {"db_path": str(db_path), "memory_root": str(root)},
            "review": {"mode": "auto-apply"},
        },
    )
    conn = open_db(db_path)
    _insert_pending(conn, root, "MEMORY.md", "index", "- [New](new.md) — added\n")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        dream,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "merge", "reason": "add link", "body": "- [Existing](existing.md) — stale\n- [New](new.md) — added\n"}]},
            provider="codex", model="gpt-5.6-luna",
        ),
        raising=False,
    )

    assert run_curate(root, db_path, config) == 0
    result = (root / "MEMORY.md").read_text(encoding="utf-8")
    assert result == "- [Existing](existing.md) — keep\n- [New](new.md) — added\n"


def test_curate_prompt_has_complete_body_and_never_reads_unsafe_target(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True, target="safe.md")
    outside = tmp_path / "outside.md"
    outside.write_text("SECRET OUTSIDE BODY", encoding="utf-8")
    conn = open_db(db_path)
    conn.execute(
        "UPDATE suggestions SET target_path='../outside.md', base_sha256=NULL, target_existed=0 WHERE id=1"
    )
    conn.commit()
    conn.close()
    captured = {}

    def fake_review(stage, prompt, schema, **kwargs):
        captured["prompt"] = prompt
        return SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "defer", "reason": "unsafe"}]},
            provider="codex", model="gpt-5.6-luna",
        )

    monkeypatch.setattr(dream, "generate", fake_review, raising=False)
    assert run_curate(root, db_path, config) == 0
    assert "SECRET OUTSIDE BODY" not in captured["prompt"]
    assert '"current_target_body": ""' in captured["prompt"]
    assert statuses(db_path)[1] == "pending"


def test_curate_prompt_snapshots_complete_current_target_body(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True)
    captured = {}

    def fake_review(stage, prompt, schema, **kwargs):
        captured["prompt"] = prompt
        return SimpleNamespace(
            output={"decisions": [
                {"suggestion_id": 1, "decision": "defer", "reason": "inspect"},
                {"suggestion_id": 2, "decision": "defer", "reason": "inspect"},
                {"suggestion_id": 3, "decision": "defer", "reason": "inspect"},
                {"suggestion_id": 4, "decision": "defer", "reason": "inspect"},
            ]},
            provider="codex", model="gpt-5.6-luna",
        )

    monkeypatch.setattr(dream, "generate", fake_review, raising=False)
    assert run_curate(root, db_path, config) == 0
    assert '"current_target_body": "original accept body"' in captured["prompt"]


def test_curate_merge_requires_string_body_before_mutation(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True, target="merge.md")
    before = _fixture_snapshot(root, db_path)
    monkeypatch.setattr(
        dream,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "merge", "reason": "missing body"}]},
            provider="codex", model="gpt-5.6-luna",
        ),
        raising=False,
    )

    assert run_curate(root, db_path, config) != 0
    assert _fixture_snapshot(root, db_path) == before
    assert statuses(db_path)[1] == "pending"
    assert not list(tmp_path.glob("memory-backups/*"))


def test_curate_merge_remove_is_safe_defer(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    root.mkdir()
    (root / "remove.md").write_text("keep", encoding="utf-8")
    db_path = tmp_path / "dream.db"
    config = load_config(
        tmp_path / "missing-config.toml",
        overrides={
            "storage": {"db_path": str(db_path), "memory_root": str(root)},
            "review": {"mode": "auto-apply"},
        },
    )
    conn = open_db(db_path)
    _insert_pending(conn, root, "remove.md", "remove", "")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        dream,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "merge", "reason": "unsupported", "body": "body"}]},
            provider="codex", model="gpt-5.6-luna",
        ),
        raising=False,
    )

    assert run_curate(root, db_path, config) == 0
    assert statuses(db_path)[1] == "pending"
    assert (root / "remove.md").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob("memory-backups/*"))


def test_curate_merge_can_refresh_an_old_conflict_after_recheck(tmp_path, monkeypatch):
    root, db_path, config = make_curation_fixture(tmp_path, auto_apply=True, target="merge.md")
    conn = open_db(db_path)
    conn.execute("UPDATE suggestions SET base_sha256='old-sha' WHERE id=1")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        dream,
        "generate",
        lambda *args, **kwargs: SimpleNamespace(
            output={"decisions": [{"suggestion_id": 1, "decision": "merge", "reason": "reconcile", "body": "merged after conflict"}]},
            provider="codex", model="gpt-5.6-luna",
        ),
        raising=False,
    )

    assert run_curate(root, db_path, config) == 0
    assert statuses(db_path)[1] == "accepted"
    assert (root / "merge.md").read_text(encoding="utf-8") == "merged after conflict"
    assert list(tmp_path.glob("memory-backups/*"))
