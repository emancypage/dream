"""Tests for review-time helpers: append-only MEMORY.md merge + DB migration.

Runnable directly (`python3 tests/test_review_helpers.py`) or via pytest.
"""

import argparse
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dream import _index_target, _merge_index, _prune_orphaned_index_lines, open_db  # noqa: E402


def test_index_target_extracts_filename():
    assert _index_target("- [Title](foo_bar.md) — hook") == "foo_bar.md"
    assert _index_target("just prose, no link") is None
    assert _index_target("## Header") is None


def test_merge_appends_only_new_targets():
    existing = "- [A](a.md) — hook a\n- [C](c.md) — hook c\n"
    proposed = "- [A](a.md) — hook a\n- [B](b.md) — hook b\n"
    merged = _merge_index(existing, proposed)
    # a.md and c.md preserved, b.md appended exactly once
    assert merged.count("(a.md)") == 1
    assert merged.count("(c.md)") == 1
    assert merged.count("(b.md)") == 1
    assert "hook b" in merged
    # existing lines stay verbatim and first
    assert merged.startswith(existing.rstrip("\n"))


def test_merge_never_reverts_existing_hook():
    # Proposed carries a STALE hook for an already-present file — must not overwrite.
    existing = "- [Deploy runbook](deploy.md) — FRESH: staging cutover done\n"
    proposed = "- [Deploy runbook](deploy.md) — STALE: cutover planned\n- [New](new.md) — n\n"
    merged = _merge_index(existing, proposed)
    assert "FRESH" in merged
    assert "STALE" not in merged
    assert "(new.md)" in merged


def test_merge_preserves_existing_only_target():
    existing = "- [Only](only.md) — keep me\n"
    proposed = "- [New](new.md) — n\n"
    merged = _merge_index(existing, proposed)
    assert "(only.md)" in merged
    assert "(new.md)" in merged


def test_merge_noop_when_all_present_returns_existing():
    existing = "- [A](a.md) — x\n- [B](b.md) — y\n"
    proposed = "- [A](a.md) — x\n- [B](b.md) — y\n"
    assert _merge_index(existing, proposed) == existing


def test_merge_into_empty_existing():
    merged = _merge_index("", "- [A](a.md) — x\n")
    assert "(a.md)" in merged


def test_merge_ignores_non_link_proposed_lines():
    existing = "- [A](a.md) — x\n"
    proposed = "## Some header\n\n- [A](a.md) — x\n"
    merged = _merge_index(existing, proposed)
    assert "## Some header" not in merged
    assert merged == existing  # nothing new to append


def test_merge_trailing_newline():
    existing = "- [A](a.md) — x\n"
    proposed = "- [B](b.md) — y\n"
    merged = _merge_index(existing, proposed)
    assert merged.endswith("\n")
    assert "\n\n" not in merged  # no blank gap introduced


def test_prune_orphaned_index_lines_drops_missing_target():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        (root / "c.md").write_text("---\nname: c\n---\nbody\n", encoding="utf-8")
        # b.md deliberately NOT created -> orphaned link
        (root / "MEMORY.md").write_text(
            "- [A](a.md) — hook a\n- [B](b.md) — hook b\n- [C](c.md) — hook c\n",
            encoding="utf-8",
        )
        changed = _prune_orphaned_index_lines(root)
        assert changed is True
        result = (root / "MEMORY.md").read_text(encoding="utf-8")
        assert "(a.md)" in result
        assert "(c.md)" in result
        assert "(b.md)" not in result


def test_prune_orphaned_index_lines_noop_when_all_targets_exist():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.md").write_text("x", encoding="utf-8")
        original = "- [A](a.md) — hook a\n"
        (root / "MEMORY.md").write_text(original, encoding="utf-8")
        changed = _prune_orphaned_index_lines(root)
        assert changed is False
        assert (root / "MEMORY.md").read_text(encoding="utf-8") == original


def test_prune_orphaned_index_lines_preserves_non_link_lines():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "MEMORY.md").write_text(
            "## Header\n\n- [Gone](gone.md) — orphaned\n\nsome prose\n",
            encoding="utf-8",
        )
        _prune_orphaned_index_lines(root)
        result = (root / "MEMORY.md").read_text(encoding="utf-8")
        assert "## Header" in result
        assert "some prose" in result
        assert "(gone.md)" not in result


def test_prune_orphaned_index_lines_no_memory_md():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        assert _prune_orphaned_index_lines(root) is False


def test_migration_adds_sug_file_to_legacy_db():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "legacy.db"
        # Simulate a pre-migration DB: suggestions table WITHOUT sug_file.
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE suggestions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
            "target_path TEXT NOT NULL, body TEXT NOT NULL, status TEXT DEFAULT 'pending')"
        )
        conn.execute("INSERT INTO suggestions(kind, target_path, body) VALUES ('new', 'x.md', 'b')")
        conn.commit()
        conn.close()

        # open_db runs schema (IF NOT EXISTS, so keeps legacy table) + _migrate.
        conn = open_db(db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(suggestions)")}
        assert "sug_file" in cols
        # legacy row survives, sug_file NULL
        row = conn.execute("SELECT sug_file FROM suggestions WHERE target_path='x.md'").fetchone()
        assert row[0] is None
        # idempotent: a second open_db must not raise
        conn.close()
        conn = open_db(db)
        conn.close()


def test_provider_migration_preserves_rows_and_seeds_legacy_ledger():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "legacy-full.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, project_slug TEXT NOT NULL, "
            "jsonl_path TEXT NOT NULL, jsonl_mtime REAL NOT NULL, started_at TEXT, ended_at TEXT, "
            "cwd TEXT, git_branch TEXT, user_msg_count INTEGER DEFAULT 0, "
            "asst_msg_count INTEGER DEFAULT 0, total_chars INTEGER DEFAULT 0, "
            "ingested_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE distilled (session_id TEXT PRIMARY KEY, distilled_at TEXT DEFAULT CURRENT_TIMESTAMP, "
            "model TEXT, input_tokens INTEGER, output_tokens INTEGER, notes_json TEXT NOT NULL, summary TEXT)"
        )
        conn.execute(
            "CREATE TABLE suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
            "target_path TEXT NOT NULL, body TEXT NOT NULL, rationale TEXT, source_sessions TEXT, "
            "status TEXT DEFAULT 'pending', reviewed_at TEXT)"
        )
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime, ended_at) "
            "VALUES ('old', 'p', '/old.jsonl', 1.5, '2026-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO distilled(session_id, model, notes_json) VALUES ('old', 'haiku', '{}')"
        )
        conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, status) VALUES ('new', 'x.md', 'x', 'pending')"
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('last_consolidate_at', '2026-02-01T00:00:00Z')"
        )
        conn.commit()
        conn.close()

        conn = open_db(db)
        session = conn.execute(
            "SELECT source, external_session_id, source_revision, parser_version FROM sessions WHERE session_id='old'"
        ).fetchone()
        assert session[0:2] == ("claude", "old")
        assert session[2].startswith("legacy:")
        assert session[3] == "claude-v1"
        distilled = conn.execute(
            "SELECT distillation_key, provider, prompt_version FROM distilled WHERE session_id='old'"
        ).fetchone()
        assert distilled == ("legacy:old", "claude", "legacy")
        assert conn.execute(
            "SELECT COUNT(*) FROM consolidated_distillations WHERE distillation_key='legacy:old'"
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE version=1").fetchone()[0] == 1
        conn.close()

        # Reopening is idempotent and does not duplicate ledger rows.
        conn = open_db(db)
        assert conn.execute("SELECT COUNT(*) FROM consolidated_distillations").fetchone()[0] == 1
        conn.close()
        backups = list(Path(d).glob("legacy-full.db.bak-provider-migration-*"))
        assert len(backups) == 1


def test_cmd_review_auto_apply_prunes_orphaned_index_lines():
    from dream import cmd_review

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "memory"
        root.mkdir()
        (root / "a.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        (root / "MEMORY.md").write_text(
            "- [A](a.md) — hook a\n- [Gone](gone.md) — orphaned already\n",
            encoding="utf-8",
        )
        db_path = Path(d) / "dream.db"
        conn = open_db(db_path)
        conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status) "
            "VALUES ('index', 'MEMORY.md', '- [A](a.md) — hook a\n', 'noop', '', 'pending')"
        )
        conn.commit()
        conn.close()

        args = argparse.Namespace(db=db_path, memory=str(root), yes=True, dry_run=False)
        cmd_review(args)

        result = (root / "MEMORY.md").read_text(encoding="utf-8")
        assert "(gone.md)" not in result
        assert "(a.md)" in result


def test_cmd_review_interactive_prunes_orphaned_index_lines(monkeypatch):
    from dream import cmd_review

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "memory"
        root.mkdir()
        (root / "a.md").write_text("---\nname: a\n---\nbody\n", encoding="utf-8")
        (root / "MEMORY.md").write_text(
            "- [A](a.md) — hook a\n- [Gone](gone.md) — orphaned already\n",
            encoding="utf-8",
        )
        db_path = Path(d) / "dream.db"
        conn = open_db(db_path)
        conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status) "
            "VALUES ('index', 'MEMORY.md', '- [A](a.md) — hook a\n', 'noop', '', 'pending')"
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr("builtins.input", lambda _: "r")  # reject the one suggestion
        args = argparse.Namespace(db=db_path, memory=str(root), yes=False, dry_run=False)
        cmd_review(args)

        result = (root / "MEMORY.md").read_text(encoding="utf-8")
        assert "(gone.md)" not in result


def test_apply_suggestion_remove_deletes_file_and_marks_accepted():
    from dream import _apply_suggestion

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "memory"
        root.mkdir()
        (root / "dead.md").write_text("---\nname: dead\n---\nbody\n", encoding="utf-8")
        db_path = Path(d) / "dream.db"
        conn = open_db(db_path)
        cur = conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status) "
            "VALUES ('remove', 'dead.md', '', 'closed', '', 'pending')"
        )
        sug_id = cur.lastrowid
        conn.commit()

        _apply_suggestion(conn, root, root / ".suggestions", sug_id, "remove", "dead.md", "", None)

        assert not (root / "dead.md").exists()
        row = conn.execute("SELECT status FROM suggestions WHERE id=?", (sug_id,)).fetchone()
        assert row[0] == "accepted"


def test_apply_suggestion_remove_refuses_path_escape():
    from dream import _apply_suggestion

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "memory"
        root.mkdir()
        outside = Path(d) / "outside.md"
        outside.write_text("do not touch\n", encoding="utf-8")
        db_path = Path(d) / "dream.db"
        conn = open_db(db_path)
        cur = conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status) "
            "VALUES ('remove', '../outside.md', '', 'malicious', '', 'pending')"
        )
        sug_id = cur.lastrowid
        conn.commit()

        _apply_suggestion(conn, root, root / ".suggestions", sug_id, "remove", "../outside.md", "", None)

        assert outside.exists()  # untouched
        row = conn.execute("SELECT status FROM suggestions WHERE id=?", (sug_id,)).fetchone()
        assert row[0] == "rejected"


@pytest.mark.parametrize("target_path", ["MEMORY.md", "./MEMORY.md", "sub/../MEMORY.md"])
def test_apply_suggestion_remove_refuses_memory_md_aliases(target_path):
    from dream import _apply_suggestion

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "memory"
        root.mkdir()
        (root / "MEMORY.md").write_text("- [A](a.md) — hook a\n", encoding="utf-8")
        db_path = Path(d) / "dream.db"
        conn = open_db(db_path)
        cur = conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status) "
            "VALUES ('remove', ?, '', 'malicious', '', 'pending')",
            (target_path,),
        )
        sug_id = cur.lastrowid
        conn.commit()

        _apply_suggestion(conn, root, root / ".suggestions", sug_id, "remove", target_path, "", None)

        assert (root / "MEMORY.md").exists()
        row = conn.execute("SELECT status FROM suggestions WHERE id=?", (sug_id,)).fetchone()
        assert row[0] == "rejected"


def test_apply_suggestion_refuses_changed_target():
    import hashlib
    from dream import _apply_suggestion

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "memory"
        root.mkdir()
        target = root / "topic.md"
        target.write_text("old\n", encoding="utf-8")
        old_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        db_path = Path(d) / "dream.db"
        conn = open_db(db_path)
        cur = conn.execute(
            "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, status, "
            "base_sha256, target_existed) VALUES ('update', 'topic.md', 'proposed', 'r', '', "
            "'pending', ?, 1)",
            (old_hash,),
        )
        sug_id = cur.lastrowid
        conn.commit()
        target.write_text("manual newer edit\n", encoding="utf-8")

        applied = _apply_suggestion(
            conn, root, root / ".suggestions", sug_id, "update", "topic.md", "proposed", None
        )

        assert applied is False
        assert target.read_text(encoding="utf-8") == "manual newer edit\n"
        assert conn.execute("SELECT status FROM suggestions WHERE id=?", (sug_id,)).fetchone()[0] == "pending"


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
