"""`patch` suggestions: the single writer, its preconditions, and every review path.

A patch replaces exactly one NFC-normalized occurrence of `find` in an NFC target
file under the same SHA precondition, protected-file checks and backups as an
`update`. Zero or several matches, or a non-NFC file, defer the suggestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

import pytest

import dream
from config import load_config
from consolidate import PatchError, patch_text, states_deletion
from dream import apply_patch, open_db


BODY = "---\nname: deploy\n---\n\nDeploys run from the build host.\nRollbacks need the on-call lead.\n"


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ------------------------------------------------------------ apply_patch


def test_apply_patch_replaces_the_single_occurrence(tmp_path):
    target = tmp_path / "deploy.md"
    target.write_text(BODY, encoding="utf-8")
    result = apply_patch(target, "the build host", "the release runner", expected_sha=_sha(target))
    assert result == BODY.replace("the build host", "the release runner")
    assert target.read_text(encoding="utf-8") == result


@pytest.mark.parametrize("find, reason", [
    ("the staging host", "matches 0 places"),
    ("the", "matches 2 places"),
])
def test_apply_patch_refuses_zero_or_several_matches(tmp_path, find, reason):
    target = tmp_path / "deploy.md"
    target.write_text(BODY, encoding="utf-8")
    with pytest.raises(PatchError, match=reason):
        apply_patch(target, find, "x", expected_sha=_sha(target))
    assert target.read_text(encoding="utf-8") == BODY


def test_apply_patch_refuses_a_changed_target(tmp_path):
    target = tmp_path / "deploy.md"
    target.write_text(BODY, encoding="utf-8")
    stale = _sha(target)
    target.write_text(BODY + "Edited by hand.\n", encoding="utf-8")
    with pytest.raises(PatchError, match="changed since"):
        apply_patch(target, "the build host", "x", expected_sha=stale)
    assert target.read_text(encoding="utf-8").endswith("Edited by hand.\n")


@pytest.mark.parametrize("expected_sha", [None, ""])
def test_apply_patch_requires_a_sha_precondition(tmp_path, expected_sha):
    target = tmp_path / "deploy.md"
    target.write_text(BODY, encoding="utf-8")
    with pytest.raises(PatchError, match="SHA"):
        apply_patch(target, "the build host", "x", expected_sha=expected_sha)


def test_apply_patch_refuses_a_missing_target(tmp_path):
    with pytest.raises(PatchError, match="does not exist"):
        apply_patch(tmp_path / "gone.md", "a", "b", expected_sha="0" * 64)


def test_apply_patch_matches_after_nfc_normalization(tmp_path):
    target = tmp_path / "cafe.md"
    composed = unicodedata.normalize("NFC", "Meet at the café on Fridays.\n")
    target.write_text(composed, encoding="utf-8")
    decomposed_find = unicodedata.normalize("NFD", "the café")
    assert decomposed_find not in composed
    result = apply_patch(target, decomposed_find, "the library", expected_sha=_sha(target))
    assert result == "Meet at the library on Fridays.\n"


def test_apply_patch_defers_a_non_nfc_target_file(tmp_path):
    target = tmp_path / "cafe.md"
    target.write_text(unicodedata.normalize("NFD", "Meet at the café.\n"), encoding="utf-8")
    before = target.read_bytes()
    with pytest.raises(PatchError, match="not NFC"):
        apply_patch(target, "Meet", "Gather", expected_sha=_sha(target))
    assert target.read_bytes() == before


def test_apply_patch_requires_a_replacement_unless_deletion_is_allowed(tmp_path):
    target = tmp_path / "deploy.md"
    target.write_text(BODY, encoding="utf-8")
    with pytest.raises(PatchError, match="empty replacement"):
        apply_patch(target, "Rollbacks need the on-call lead.\n", "", expected_sha=_sha(target))
    result = apply_patch(target, "Rollbacks need the on-call lead.\n", "", expected_sha=_sha(target), allow_empty=True)
    assert "Rollbacks" not in result


def test_patch_text_rejects_an_empty_find():
    with pytest.raises(PatchError, match="empty"):
        patch_text("abc", "", "x")


@pytest.mark.parametrize("rationale, expected", [
    ("Delete the obsolete rollback rule", True),
    ("deletion: the on-call lead role was removed", True),
    ("Remove the stale sentence about staging", True),
    ("Rename the host", False),
    ("Clarify the wording", False),
])
def test_states_deletion(rationale, expected):
    assert states_deletion(rationale) is expected


# ------------------------------------------------------------ review paths


def _fixture(tmp_path, *, find="the build host", body="the release runner", target="deploy.md",
             rationale="host renamed", auto_apply=True, content=BODY):
    root = tmp_path / "memory"
    root.mkdir()
    path = root / target
    if content is not None:
        path.write_text(content, encoding="utf-8")
    db_path = tmp_path / "dream.db"
    config = load_config(tmp_path / "missing-config.toml", overrides={
        "storage": {"db_path": str(db_path), "memory_root": str(root)},
        "review": {"mode": "auto-apply" if auto_apply else "suggest-only"},
    })
    conn = open_db(db_path)
    sug_dir = root / ".suggestions"
    sug_dir.mkdir()
    (sug_dir / "preview-patch.md").write_text("preview", encoding="utf-8")
    conn.execute(
        "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status, sug_file, "
        "base_sha256, target_existed, find_text) VALUES ('patch', ?, ?, ?, 's-1', 'pending', "
        "'preview-patch.md', ?, ?, ?)",
        (target, body, rationale, _sha(path), int(path.exists()), find),
    )
    conn.commit()
    conn.close()
    return root, db_path, config


def _suggestions(root, db_path, config, command, **extra):
    return dream.cmd_suggestions(argparse.Namespace(
        db=db_path, memory=str(root), suggestions_cmd=command, config=config, **extra))


def _row(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT status, kind, find_text FROM suggestions WHERE id=1").fetchone()
    finally:
        conn.close()


def test_accept_applies_the_patch_with_a_backup_and_drops_the_preview(tmp_path):
    root, db_path, config = _fixture(tmp_path)
    assert _suggestions(root, db_path, config, "accept", id=1) == 0
    assert (root / "deploy.md").read_text(encoding="utf-8") == BODY.replace("the build host", "the release runner")
    assert _row(db_path)[0] == "accepted"
    assert not (root / ".suggestions" / "preview-patch.md").exists()
    [backup] = list(tmp_path.glob("memory-backups/*"))
    assert (backup / "deploy.md").read_text(encoding="utf-8") == BODY


@pytest.mark.parametrize("find, reason", [("missing text", "matches 0 places"), ("the", "matches 2 places")])
def test_accept_defers_an_ambiguous_patch_and_reports_why(tmp_path, capsys, find, reason):
    root, db_path, config = _fixture(tmp_path, find=find)
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert (root / "deploy.md").read_text(encoding="utf-8") == BODY
    assert _row(db_path)[0] == "pending"
    assert (root / ".suggestions" / "preview-patch.md").exists()
    out = capsys.readouterr().out
    assert "deferred #1:" in out and reason in out


def test_accept_refuses_a_patch_whose_target_changed(tmp_path, capsys):
    root, db_path, config = _fixture(tmp_path)
    (root / "deploy.md").write_text(BODY + "Edited by hand.\n", encoding="utf-8")
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert (root / "deploy.md").read_text(encoding="utf-8").endswith("Edited by hand.\n")
    assert _row(db_path)[0] == "pending"
    assert "conflict #1" in capsys.readouterr().out


def test_accept_never_patches_memory_md(tmp_path):
    index = "- [Deploy](deploy.md) — the build host\n"
    root, db_path, config = _fixture(tmp_path, target="MEMORY.md", content=index,
                                     find="the build host", body="the runner")
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert (root / "MEMORY.md").read_text(encoding="utf-8") == index
    assert _row(db_path)[0] == "rejected"


def test_accept_refuses_a_patch_of_the_retention_policy(tmp_path):
    policy = json.dumps({"instructions": "keep the build host private"})
    root, db_path, config = _fixture(tmp_path, target=".dream-retention.json", content=policy,
                                     find="private", body="public")
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert (root / ".dream-retention.json").read_text(encoding="utf-8") == policy
    assert _row(db_path)[0] == "rejected"


def test_accept_refuses_a_patch_outside_the_memory_root(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text(BODY, encoding="utf-8")
    root, db_path, config = _fixture(tmp_path, target="../outside.md", content=None)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE suggestions SET base_sha256=?, target_existed=1 WHERE id=1", (_sha(outside),))
    conn.commit()
    conn.close()
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert outside.read_text(encoding="utf-8") == BODY
    assert _row(db_path)[0] == "rejected"


def test_accept_rejects_a_patch_that_introduces_retention_excluded_text(tmp_path):
    root, db_path, config = _fixture(tmp_path, body="the SECRET-PROJECT runner")
    (root / ".dream-retention.json").write_text(json.dumps({"text_markers": ["SECRET-PROJECT"]}), encoding="utf-8")
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert (root / "deploy.md").read_text(encoding="utf-8") == BODY
    assert _row(db_path)[0] == "rejected"


def test_accept_allows_an_empty_replacement_only_for_a_stated_deletion(tmp_path, capsys):
    root, db_path, config = _fixture(tmp_path, find="Rollbacks need the on-call lead.\n", body="",
                                     rationale="tighten wording")
    assert _suggestions(root, db_path, config, "accept", id=1) == 3
    assert _row(db_path)[0] == "pending"
    assert "empty replacement" in capsys.readouterr().out

    tmp2 = tmp_path / "second"
    tmp2.mkdir()
    root, db_path, config = _fixture(tmp2, find="Rollbacks need the on-call lead.\n", body="",
                                     rationale="Delete the obsolete rollback rule")
    assert _suggestions(root, db_path, config, "accept", id=1) == 0
    assert "Rollbacks" not in (root / "deploy.md").read_text(encoding="utf-8")


def test_merge_turns_a_patch_into_a_full_body_update(tmp_path):
    root, db_path, config = _fixture(tmp_path, find="the")
    merged = tmp_path / "merged.md"
    merged.write_text("---\nname: deploy\n---\n\nDeploys run from the release runner.\n", encoding="utf-8")
    assert _suggestions(root, db_path, config, "merge", id=1, body_file=str(merged)) == 0
    assert (root / "deploy.md").read_text(encoding="utf-8") == merged.read_text(encoding="utf-8")
    assert _row(db_path) == ("accepted", "update", None)
    assert list(tmp_path.glob("memory-backups/*"))


def test_apply_all_handles_patches(tmp_path):
    root, db_path, config = _fixture(tmp_path)
    assert _suggestions(root, db_path, config, "apply-all") == 0
    assert "the release runner" in (root / "deploy.md").read_text(encoding="utf-8")
    assert _row(db_path)[0] == "accepted"


def test_apply_all_reports_a_deferred_patch_as_not_applied(tmp_path):
    root, db_path, config = _fixture(tmp_path, find="the")
    assert _suggestions(root, db_path, config, "apply-all") == 3
    assert _row(db_path)[0] == "pending"


def test_list_includes_the_find_text(tmp_path, capsys):
    root, db_path, config = _fixture(tmp_path)
    assert _suggestions(root, db_path, config, "list") == 0
    [item] = json.loads(capsys.readouterr().out)["suggestions"]
    assert item["kind"] == "patch"
    assert item["find"] == "the build host"
    assert item["conflict"] is False


def test_review_yes_applies_a_patch(tmp_path):
    root, db_path, config = _fixture(tmp_path)
    args = argparse.Namespace(db=db_path, memory=str(root), config=config, yes=True, dry_run=False)
    assert dream.cmd_review(args) == 0
    assert "the release runner" in (root / "deploy.md").read_text(encoding="utf-8")
    assert _row(db_path)[0] == "accepted"


def test_interactive_review_shows_the_patched_diff(tmp_path, monkeypatch, capsys):
    root, db_path, config = _fixture(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "s")
    args = argparse.Namespace(db=db_path, memory=str(root), config=config, yes=False, dry_run=False)
    assert dream.cmd_review(args) == 0
    out = capsys.readouterr().out
    assert "-Deploys run from the build host." in out
    assert "+Deploys run from the release runner." in out
    assert "find: the build host" in out
    assert _row(db_path)[0] == "pending"


def test_interactive_review_explains_a_patch_that_cannot_apply(tmp_path, monkeypatch, capsys):
    root, db_path, config = _fixture(tmp_path, find="the")
    monkeypatch.setattr("builtins.input", lambda _prompt: "a")
    args = argparse.Namespace(db=db_path, memory=str(root), config=config, yes=False, dry_run=False)
    assert dream.cmd_review(args) == 0
    out = capsys.readouterr().out
    assert "cannot apply" in out
    assert "deferred #1:" in out
    assert (root / "deploy.md").read_text(encoding="utf-8") == BODY
    assert _row(db_path)[0] == "pending"


# ------------------------------------------------------------ migration


def test_migration_backs_up_once_then_adds_find_text_to_a_pre_patch_db(tmp_path):
    db_path = tmp_path / "dream.db"
    conn = open_db(db_path)
    conn.execute("DROP TABLE suggestions")
    conn.execute(
        "CREATE TABLE suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, "
        "kind TEXT NOT NULL, target_path TEXT NOT NULL, body TEXT NOT NULL, rationale TEXT, source_sessions TEXT, "
        "status TEXT DEFAULT 'pending', reviewed_at TEXT, sug_file TEXT, base_sha256 TEXT, target_existed INTEGER, "
        "consolidation_run_id TEXT)"
    )
    conn.execute("INSERT INTO suggestions(kind, target_path, body) VALUES ('update', 'x.md', 'b')")
    conn.commit()
    conn.close()

    conn = open_db(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(suggestions)")}
    assert "find_text" in columns
    assert conn.execute("SELECT kind, find_text FROM suggestions WHERE target_path='x.md'").fetchone() == (
        "update", None)
    conn.close()
    [backup] = list(tmp_path.glob("dream.db.bak-provider-migration-*"))
    legacy = sqlite3.connect(backup)
    try:
        assert "find_text" not in {row[1] for row in legacy.execute("PRAGMA table_info(suggestions)")}
    finally:
        legacy.close()

    open_db(db_path).close()
    assert list(tmp_path.glob("dream.db.bak-provider-migration-*")) == [backup]


def test_pre_patch_db_reads_find_text_as_none_without_migrating(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE suggestions (id INTEGER PRIMARY KEY, kind TEXT, target_path TEXT, body TEXT)")
    conn.execute("INSERT INTO suggestions(kind, target_path, body) VALUES ('update', 'x.md', 'b')")
    conn.commit()
    assert dream._patch_fields(conn, 1) == (None, "")
    assert {row[1] for row in conn.execute("PRAGMA table_info(suggestions)")} == {"id", "kind", "target_path", "body"}
    conn.close()
