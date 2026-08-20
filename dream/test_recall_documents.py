"""Task 3: recall document synchronization.

Covers stable UUIDv5 document IDs, the SyncReport counters, approved-memory /
distilled / raw-transcript source fields, disabled raw-transcript collection,
updates and stale deletion, atomic rollback after collection and write/FTS
failures, FTS repair, silent (no-stdout) operation, and that existing ingest
behavior remains unchanged.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

import dream as dream_cli
import ingest as ingest_mod
import recall_documents
from recall_documents import SyncReport, stable_document_id, synchronize_recall_documents
from sources.base import Message, ParsedSession


ROOT = Path(__file__).parent
SCHEMA = (ROOT / "schema.sql").read_text(encoding="utf-8")

NOW = dt.datetime(2026, 8, 20, 12, 0, 0, tzinfo=dt.timezone.utc)
NOW_ISO = "2026-08-20T12:00:00+00:00"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _db(tmp_path: Path, name: str = "dream.db") -> sqlite3.Connection:
    path = tmp_path / name
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _memory_root(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    root = tmp_path / "memory"
    root.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _memory_rows(conn: sqlite3.Connection) -> dict[str, tuple]:
    return {
        row[0]: row
        for row in conn.execute(
            "SELECT id, content_sha256, source_kind, trust_level, project_slug, "
            "source_path, source_updated_at, indexed_at, source_version, text "
            "FROM recall_documents ORDER BY id"
        )
    }


def _add_transcript(
    conn: sqlite3.Connection,
    session_id: str = "codex:tx-1",
    source: str = "codex",
    external_session_id: str = "tx-1",
    project_slug: str = "-home-user-project",
    revision: str = "rev-1",
    messages: list[tuple[int, str, str | None, str]] | None = None,
    started_at: str | None = "2026-08-10T09:00:00Z",
) -> None:
    if messages is None:
        messages = [
            (0, "user", "2026-08-10T09:00:01Z", "user question one"),
            (1, "assistant", "2026-08-10T09:00:05Z", "assistant answer one"),
            (2, "user", None, "user question two"),
            (3, "assistant", None, "assistant answer two"),
        ]
    conn.execute(
        "INSERT INTO sessions(session_id, source, external_session_id, source_revision, "
        "parser_version, project_slug, jsonl_path, jsonl_mtime, started_at) "
        "VALUES (?, ?, ?, ?, 'codex-jsonl-v1', ?, ?, 1.0, ?)",
        (session_id, source, external_session_id, revision, project_slug,
         f"/sessions/{session_id}.jsonl", started_at),
    )
    conn.executemany(
        "INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES (?, ?, ?, ?, ?)",
        [(session_id, seq, role, timestamp, text)
         for seq, role, timestamp, text in messages],
    )
    conn.commit()


def _add_distilled(
    conn: sqlite3.Connection,
    session_id: str = "codex:tx-1",
    key: str = "key-1",
    notes: dict | None = None,
    summary: str = "short prose summary",
    project_slug: str = "-home-user-project",
    distilled_at: str = "2026-08-10T10:00:00Z",
) -> None:
    conn.execute(
        "INSERT INTO sessions(session_id, source, external_session_id, source_revision, "
        "parser_version, project_slug, jsonl_path, jsonl_mtime) "
        "VALUES (?, 'codex', ?, ?, 'codex-jsonl-v1', ?, ?, 1.0)",
        (session_id, session_id.split(":", 1)[1], "rev-x", project_slug,
         f"/sessions/{session_id}.jsonl"),
    )
    conn.execute(
        "INSERT INTO distilled(session_id, distillation_key, notes_json, summary, "
        "distilled_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, key,
         json.dumps(notes if notes is not None else {"facts": ["note one"]},
                    ensure_ascii=False),
         summary, distilled_at),
    )
    conn.commit()


# ----------------------------------------------------------- stable IDs


def test_stable_document_id_is_a_deterministic_uuidv5():
    doc_id = stable_document_id("approved_memory", "memory:notes/api.md")
    assert isinstance(doc_id, str)
    parsed = uuid.UUID(doc_id)
    assert parsed.version == 5
    assert doc_id == stable_document_id("approved_memory", "memory:notes/api.md")


def test_stable_document_id_uses_one_fixed_namespace_for_every_kind():
    memory = stable_document_id("approved_memory", "memory:one.md")
    distilled = stable_document_id("distilled_summary", "distilled:key-1")
    transcript = stable_document_id("raw_transcript", "transcript:codex:s1")
    # One shared fixed namespace: all three IDs must be UUIDv5 under the same
    # namespace UUID, reconstructed from kind + locator.
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, "dream-recall-documents")
    assert memory == str(uuid.uuid5(namespace, f"approved_memory:memory:one.md"))
    assert distilled == str(uuid.uuid5(namespace, f"distilled_summary:distilled:key-1"))
    assert transcript == str(uuid.uuid5(namespace, f"raw_transcript:transcript:codex:s1"))


def test_stable_document_id_differs_across_kinds_and_locators():
    a = stable_document_id("approved_memory", "memory:one.md")
    b = stable_document_id("approved_memory", "memory:two.md")
    c = stable_document_id("distilled_summary", "memory:one.md")
    assert a != b
    assert a != c
    assert b != c
    for value in (a, b, c):
        assert uuid.UUID(value).version == 5


# ------------------------------------------------------------- SyncReport


def test_sync_report_exposes_all_counts():
    report = SyncReport(scanned=1, inserted=2, updated=3, deleted=4,
                        skipped=5, raw_included=6)
    assert (report.scanned, report.inserted, report.updated, report.deleted,
            report.skipped, report.raw_included) == (1, 2, 3, 4, 5, 6)


# --------------------------------------------- approved memory fields


def test_approved_memory_source_fields_and_sha256(tmp_path):
    conn = _db(tmp_path)
    content = "Approved memory: deploy postgres migrations with pgbouncer."
    root = _memory_root(tmp_path, {"notes/api.md": content})

    report = synchronize_recall_documents(
        conn, root, include_raw_transcripts=True, now=NOW
    )

    rows = _memory_rows(conn)
    doc_id = stable_document_id("approved_memory", "memory:notes/api.md")
    assert set(rows) == {doc_id}
    row = rows[doc_id]
    (rid, sha, kind, trust, project, source_path, source_updated_at,
     indexed_at, version, text) = row
    assert rid == doc_id
    assert sha == _sha(content)
    assert kind == "approved_memory"
    assert trust == "user_approved"
    assert project is None
    assert source_path == "notes/api.md"
    assert source_updated_at == NOW_ISO
    assert indexed_at == NOW_ISO
    assert version == _sha(content)
    assert text == content
    assert report.inserted == 1
    conn.close()


def test_memory_collection_skips_hidden_transient_and_special_directories(tmp_path):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {
        "keep.md": "kept memory",
        "MEMORY.md": "index memory",
        ".hidden.md": "hidden memory",
        "notes/keep-sub.md": "sub memory",
        "notes/.hidden-sub.md": "hidden sub memory",
        "notes/tmp.md.tmp": "transient file",
        ".suggestions/preview.md": "suggestion preview",
        "memory-backups/old.md": "backup copy",
    })

    report = synchronize_recall_documents(conn, root, include_raw_transcripts=False)

    rows = _memory_rows(conn)
    ids = {
        stable_document_id("approved_memory", "memory:keep.md"),
        stable_document_id("approved_memory", "memory:MEMORY.md"),
        stable_document_id("approved_memory", "memory:notes/keep-sub.md"),
    }
    assert set(rows) == ids
    assert all(row[5] in {"keep.md", "MEMORY.md", "notes/keep-sub.md"}
               for row in rows.values())
    assert report.inserted == 3
    assert report.raw_included == 0
    conn.close()


def test_memory_collection_tolerates_missing_root(tmp_path):
    conn = _db(tmp_path)
    report = synchronize_recall_documents(
        conn, tmp_path / "absent", include_raw_transcripts=False, now=NOW
    )
    assert report == SyncReport(0, 0, 0, 0, 0, 0)
    assert conn.execute("SELECT COUNT(*) FROM recall_documents").fetchone()[0] == 0
    conn.close()


# --------------------------------------------------- distilled fields


def test_distilled_source_fields(tmp_path):
    conn = _db(tmp_path)
    _add_distilled(
        conn,
        session_id="codex:tx-1",
        key="key-1",
        notes={"facts": ["note one"], "preferences": []},
        summary="short prose summary",
        project_slug="-home-user-project",
        distilled_at="2026-08-10T10:00:00Z",
    )

    report = synchronize_recall_documents(conn, _memory_root(tmp_path),
                                          include_raw_transcripts=False, now=NOW)

    doc_id = stable_document_id("distilled_summary", "distilled:key-1")
    row = conn.execute(
        "SELECT content_sha256, source_kind, trust_level, project_slug, source_path, "
        "source_updated_at, indexed_at, source_version, text "
        "FROM recall_documents WHERE id=?",
        (doc_id,),
    ).fetchone()
    assert row is not None
    sha, kind, trust, project, source_path, source_updated_at, indexed_at, version, text = row
    assert kind == "distilled_summary"
    assert trust == "model_distilled"
    assert project == "-home-user-project"
    assert source_path == "distilled:key-1"
    assert source_updated_at == "2026-08-10T10:00:00Z"
    assert indexed_at == NOW_ISO
    assert version == "key-1"
    assert text == json.dumps({"facts": ["note one"], "preferences": []}, ensure_ascii=False)
    assert sha == _sha(text)
    assert report.inserted == 1
    conn.close()


def test_distilled_text_falls_back_to_summary_when_notes_json_is_empty(tmp_path):
    conn = _db(tmp_path)
    _add_distilled(conn, session_id="codex:tx-1", key="key-2",
                   notes={}, summary="fallback prose")
    synchronize_recall_documents(conn, _memory_root(tmp_path),
                                 include_raw_transcripts=False, now=NOW)
    doc_id = stable_document_id("distilled_summary", "distilled:key-2")
    text = conn.execute(
        "SELECT text FROM recall_documents WHERE id=?", (doc_id,)
    ).fetchone()[0]
    assert text == "fallback prose"
    conn.close()


# --------------------------------------------------- raw transcript fields


def test_raw_transcript_source_fields(tmp_path):
    conn = _db(tmp_path)
    _add_transcript(conn)

    report = synchronize_recall_documents(conn, _memory_root(tmp_path),
                                          include_raw_transcripts=True, now=NOW)

    doc_id = stable_document_id("raw_transcript", "transcript:codex:tx-1")
    row = conn.execute(
        "SELECT content_sha256, source_kind, trust_level, project_slug, source_path, "
        "source_updated_at, indexed_at, source_version, text "
        "FROM recall_documents WHERE id=?",
        (doc_id,),
    ).fetchone()
    assert row is not None
    sha, kind, trust, project, source_path, source_updated_at, indexed_at, version, text = row
    assert kind == "raw_transcript"
    assert trust == "untrusted_transcript"
    assert project == "-home-user-project"
    assert source_path == "transcript:codex:tx-1"
    assert source_updated_at == "2026-08-10T09:00:00Z"
    assert indexed_at == NOW_ISO
    assert version == "rev-1"
    assert "USER" in text and "ASSISTANT" in text
    assert "2026-08-10T09:00:01Z" in text
    assert "user question one" in text
    assert "assistant answer two" in text
    assert "user question two" in text
    assert sha == _sha(text)
    assert report.raw_included == 1
    assert report.inserted == 1
    conn.close()


def test_raw_transcript_messages_are_ordered_by_sequence(tmp_path):
    conn = _db(tmp_path)
    _add_transcript(
        conn,
        messages=[
            (5, "user", None, "later user line"),
            (1, "assistant", None, "earlier assistant line"),
        ],
    )
    synchronize_recall_documents(conn, _memory_root(tmp_path),
                                 include_raw_transcripts=True, now=NOW)
    doc_id = stable_document_id("raw_transcript", "transcript:codex:tx-1")
    text = conn.execute(
        "SELECT text FROM recall_documents WHERE id=?", (doc_id,)
    ).fetchone()[0]
    assert text.index("earlier assistant line") < text.index("later user line")
    conn.close()


def test_disabled_raw_transcript_collection_inserts_nothing_and_reports_zero(tmp_path):
    conn = _db(tmp_path)
    _add_transcript(conn)

    report = synchronize_recall_documents(conn, _memory_root(tmp_path),
                                          include_raw_transcripts=False, now=NOW)

    assert report.raw_included == 0
    assert report.inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM recall_documents").fetchone()[0] == 0
    conn.close()


# --------------------------------------------- updates and stale deletion


def test_resync_updates_changed_content_deletes_stale_and_counts_skipped(tmp_path):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {
        "notes/api.md": "original content",
        "notes/gone.md": "will be deleted",
    })
    first = synchronize_recall_documents(conn, root,
                                         include_raw_transcripts=False, now=NOW)
    assert first.inserted == 2 and first.updated == 0 and first.deleted == 0

    root.joinpath("notes/gone.md").unlink()
    root.joinpath("notes/api.md").write_text("updated content", encoding="utf-8")
    root.joinpath("notes/new.md").write_text("brand new content", encoding="utf-8")

    second = synchronize_recall_documents(conn, root,
                                          include_raw_transcripts=False, now=NOW)

    assert second.inserted == 1
    assert second.updated == 1
    assert second.deleted == 1
    api_id = stable_document_id("approved_memory", "memory:notes/api.md")
    row = conn.execute(
        "SELECT text, source_version FROM recall_documents WHERE id=?", (api_id,)
    ).fetchone()
    assert row[0] == "updated content"
    assert row[1] == _sha("updated content")
    gone_id = stable_document_id("approved_memory", "memory:notes/gone.md")
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents WHERE id=?", (gone_id,)
    ).fetchone()[0] == 0
    new_id = stable_document_id("approved_memory", "memory:notes/new.md")
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents WHERE id=?", (new_id,)
    ).fetchone()[0] == 1

    third = synchronize_recall_documents(conn, root,
                                         include_raw_transcripts=False, now=NOW)
    assert third.inserted == 0 and third.updated == 0 and third.deleted == 0
    assert third.skipped == 2
    conn.close()


def test_stale_rows_are_never_deleted_when_a_source_kind_collection_fails(tmp_path,
                                                                          monkeypatch):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "original content"})
    _add_transcript(conn, revision="rev-1")
    synchronize_recall_documents(conn, root, include_raw_transcripts=True, now=NOW)
    transcript_id = stable_document_id("raw_transcript", "transcript:codex:tx-1")
    api_id = stable_document_id("approved_memory", "memory:notes/api.md")
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents WHERE id IN (?, ?)",
        (transcript_id, api_id),
    ).fetchone()[0] == 2

    called_with = {}

    def _boom(conn, memory_root, *, include_raw_transcripts=False):
        called_with["root"] = memory_root
        raise OSError("transcript collection exploded")

    monkeypatch.setattr(dream_cli, "synchronize_recall_documents", _boom)

    monkeypatch.setattr(dream_cli, "ingest_configured_sources",
                        lambda conn, config=None, selected=None, force=False:
                        [("codex", 0, 0, 0)])
    root.joinpath("notes/api.md").unlink()  # stale once the guarded sync fails

    from config import DreamConfig, _read_toml, _merge

    config_data = _merge(
        _read_toml(ROOT / "default-config.toml"),
        {"storage": {"db_path": str(tmp_path / "dream.db"),
                     "memory_root": str(root)}},
    )
    args = type("Args", (), {"db": tmp_path / "dream.db", "projects": None,
                             "source": None, "force": False,
                             "config": DreamConfig(data=config_data, path=None)})()
    rc = dream_cli.cmd_ingest(args)
    assert rc == 0  # command still succeeds; previous complete state retained
    assert str(called_with["root"]) == str(root)  # sync saw the config memory_root

    with sqlite3.connect(tmp_path / "dream.db") as c:
        rows = c.execute("SELECT id FROM recall_documents ORDER BY id").fetchall()
    assert [row[0] for row in rows] == [api_id, transcript_id]
    conn.close()


# ------------------------------------------------------------ atomicity


def test_collection_failure_rolls_back_and_preserves_previous_state(tmp_path):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "original content"})
    synchronize_recall_documents(conn, root,
                                 include_raw_transcripts=False, now=NOW)
    before = conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()

    class _BoomPath:
        """A narrow Path-like facade that forwards to a real Path but fails
        on ``is_dir`` so the collection step raises before any write."""

        def __init__(self, real: Path):
            self._real = real

        def is_dir(self):
            raise OSError("boom at the start of the walk")

        def __fspath__(self):
            return str(self._real)

        def __str__(self):
            return str(self._real)

        def __repr__(self):
            return f"_BoomPath({self._real!r})"

    with pytest.raises(OSError):
        synchronize_recall_documents(
            conn,
            _BoomPath(root),  # type: ignore[arg-type]
            include_raw_transcripts=False, now=NOW,
        )
    after = conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()
    assert after == before
    conn.close()


def test_write_failure_rolls_back_a_fully_prepared_update(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "original content"})
    first = synchronize_recall_documents(conn, root,
                                         include_raw_transcripts=False, now=NOW)
    assert first.inserted == 1
    before = conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()

    root.joinpath("notes/api.md").write_text("updated content", encoding="utf-8")

    def _explode_upsert(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk full during upsert")

    monkeypatch.setattr(recall_documents, "_upsert_document", _explode_upsert)
    with pytest.raises(sqlite3.OperationalError):
        synchronize_recall_documents(conn, root,
                                     include_raw_transcripts=False, now=NOW)
    after = conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()
    assert after == before
    conn.close()


def test_fts_failure_rolls_back_a_fully_prepared_write(tmp_path, monkeypatch):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "original content"})
    first = synchronize_recall_documents(conn, root,
                                         include_raw_transcripts=False, now=NOW)
    assert first.inserted == 1
    before = conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()
    root.joinpath("notes/api.md").write_text("updated content", encoding="utf-8")

    def _explode_rebuild(*_args, **_kwargs):
        raise sqlite3.OperationalError("fts shadow page corruption")

    monkeypatch.setattr(dream_cli, "rebuild_recall_fts", _explode_rebuild)
    with pytest.raises(sqlite3.OperationalError):
        synchronize_recall_documents(conn, root,
                                     include_raw_transcripts=False, now=NOW)
    after = conn.execute("SELECT * FROM recall_documents ORDER BY id").fetchall()
    assert after == before
    conn.close()


def test_missing_fts_index_is_rebuilt_inside_synchronization(tmp_path):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "recall fts marker text"})
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, now=NOW)
    conn.execute("DELETE FROM recall_documents_fts")
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts "
        "WHERE recall_documents_fts MATCH 'marker'"
    ).fetchone()[0] == 0

    root.joinpath("notes/api.md").write_text("recall fts marker text again",
                                              encoding="utf-8")
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, now=NOW)
    hits = conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts "
        "WHERE recall_documents_fts MATCH 'marker'"
    ).fetchone()[0]
    assert hits == 1
    conn.close()


def test_corrupt_fts_shadow_data_is_repaired_inside_synchronization(tmp_path):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "first recall words"})
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, now=NOW)
    # Simulate lost/corrupt shadow data: keep the FTS5 table registered but empty.
    conn.execute("DELETE FROM recall_documents_fts")
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts "
        "WHERE recall_documents_fts MATCH 'first'"
    ).fetchone()[0] == 0

    root.joinpath("notes/api.md").write_text("second recall words", encoding="utf-8")
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, now=NOW)
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts "
        "WHERE recall_documents_fts MATCH 'second'"
    ).fetchone()[0] == 1
    conn.close()


# ---------------------------------------------------------------- stdout


def test_synchronization_never_prints_to_stdout(tmp_path, capsys):
    conn = _db(tmp_path)
    root = _memory_root(tmp_path, {"notes/api.md": "silent sync content"})
    _add_transcript(conn)
    synchronize_recall_documents(conn, root, include_raw_transcripts=True, now=NOW)
    synchronize_recall_documents(conn, root, include_raw_transcripts=False, now=NOW)
    captured = capsys.readouterr()
    assert captured.out == ""
    conn.close()


# ------------------------------------------------------- ingest behavior


@contextlib.contextmanager
def _patch_module_attr(module, name, func):
    original = getattr(module, name)
    setattr(module, name, func)
    try:
        yield
    finally:
        setattr(module, name, original)


def test_ingest_command_keeps_printing_summary_and_synchronizes(tmp_path,
                                                                capsys, monkeypatch):
    root = _memory_root(tmp_path, {"notes/api.md": "memory after ingest"})

    db_path = tmp_path / "dream.db"
    from config import DreamConfig, _read_toml, _merge

    config_data = _merge(
        _read_toml(ROOT / "default-config.toml"),
        {"storage": {"db_path": str(db_path), "memory_root": str(root)}},
    )
    config = DreamConfig(data=config_data, path=None)

    args = type("Args", (), {"db": db_path, "projects": None,
                             "source": None, "force": False,
                             "config": config})()

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")
    parsed = ParsedSession(
        source="codex",
        external_session_id="session-1",
        revision="test-revision-1",
        parser_version="codex-jsonl-v1",
        path=transcript,
        project_slug="-home-user-project",
        started_at="2026-08-08T10:00:00Z",
        ended_at="2026-08-08T10:01:00Z",
        cwd="/home/user/project",
        git_branch=None,
        messages=[Message(1, "user", "2026-08-08T10:01:00Z", "hello world")],
    )

    def _fake_configured_sources(conn, config=None, selected=None, force=False):
        changed, count = ingest_mod.ingest_parsed_session(parsed, conn, force=force)
        return [("codex", 1, 1 if changed else 0, count if changed else 0)]

    with _patch_module_attr(dream_cli, "ingest_configured_sources",
                            _fake_configured_sources):
        rc = dream_cli.cmd_ingest(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "codex: scanned 1, ingested 1, 1 messages stored." in out
    assert f"DB: {db_path}" in out
    with sqlite3.connect(db_path) as c:
        doc_id = stable_document_id("approved_memory", "memory:notes/api.md")
        assert c.execute(
            "SELECT COUNT(*) FROM recall_documents WHERE id=?", (doc_id,)
        ).fetchone()[0] == 1
        assert c.execute(
            "SELECT COUNT(*) FROM recall_documents WHERE source_kind='raw_transcript'"
        ).fetchone()[0] == 0
