"""Task 2: recall schema and migration.

Covers recall table creation (columns, primary keys, CHECK constraints),
the external-content FTS5 index with `unicode61 remove_diacritics 2`,
its insert/delete triggers, the rebuild function, the version-2 migration
marker, and preservation of legacy transcript rows (messages +
messages_fts must survive migration untouched).
"""

import sqlite3
from pathlib import Path

import pytest

from dream import open_db, rebuild_recall_fts


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, dict]:
    return {
        row[1]: {"notnull": row[3], "pk": row[5]}
        for row in conn.execute(f"PRAGMA table_info({table})")
    }


def _legacy_schema() -> str:
    """v0.0.1 schema: everything except the recall objects.

    Faithful to the released schema — sessions/distilled/suggestions carry the
    full post-provider-migration column set, so no ALTER/backup path fires;
    the point is that legacy transcript rows and messages_fts survive.
    """
    return """
    CREATE TABLE sessions (
        session_id      TEXT PRIMARY KEY,
        source          TEXT NOT NULL DEFAULT 'claude',
        external_session_id TEXT,
        source_revision TEXT,
        parser_version  TEXT,
        project_slug    TEXT NOT NULL,
        jsonl_path      TEXT NOT NULL,
        jsonl_mtime     REAL NOT NULL,
        started_at      TEXT,
        ended_at        TEXT,
        cwd             TEXT,
        git_branch      TEXT,
        user_msg_count  INTEGER DEFAULT 0,
        asst_msg_count  INTEGER DEFAULT 0,
        total_chars     INTEGER DEFAULT 0,
        ingested_at     TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX idx_sessions_project ON sessions(project_slug);
    CREATE INDEX idx_sessions_started ON sessions(started_at);
    CREATE TABLE messages (
        session_id   TEXT NOT NULL,
        seq          INTEGER NOT NULL,
        role         TEXT NOT NULL,
        timestamp    TEXT,
        text         TEXT NOT NULL,
        PRIMARY KEY (session_id, seq),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    CREATE VIRTUAL TABLE messages_fts USING fts5(
        text,
        role UNINDEXED,
        session_id UNINDEXED,
        timestamp UNINDEXED,
        content='messages',
        content_rowid='rowid',
        tokenize='porter unicode61'
    );
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, text, role, session_id, timestamp)
        VALUES (new.rowid, new.text, new.role, new.session_id, new.timestamp);
    END;
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text, role, session_id, timestamp)
        VALUES('delete', old.rowid, old.text, old.role, old.session_id, old.timestamp);
    END;
    CREATE TABLE distilled (
        session_id    TEXT PRIMARY KEY,
        distillation_key TEXT,
        distilled_at  TEXT DEFAULT CURRENT_TIMESTAMP,
        provider      TEXT,
        model         TEXT,
        source_revision TEXT,
        parser_version TEXT,
        prompt_version TEXT,
        provider_options TEXT,
        input_tokens  INTEGER,
        output_tokens INTEGER,
        usage_json    TEXT,
        duration_ms   INTEGER,
        notes_json    TEXT NOT NULL,
        summary       TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
    );
    CREATE TABLE suggestions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
        kind          TEXT NOT NULL,
        target_path   TEXT NOT NULL,
        body          TEXT NOT NULL,
        rationale     TEXT,
        source_sessions TEXT,
        status        TEXT DEFAULT 'pending',
        reviewed_at   TEXT,
        sug_file      TEXT,
        base_sha256   TEXT,
        target_existed INTEGER,
        consolidation_run_id TEXT
    );
    CREATE TABLE meta (
        key    TEXT PRIMARY KEY,
        value  TEXT
    );
    CREATE TABLE consolidation_runs (
        run_id        TEXT PRIMARY KEY,
        provider      TEXT,
        model         TEXT,
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE consolidated_distillations (
        distillation_key TEXT PRIMARY KEY,
        session_id       TEXT NOT NULL,
        run_id           TEXT NOT NULL,
        consolidated_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """


def _make_legacy_db(tmp_path: Path, name: str = "legacy.db") -> Path:
    """A v0.0.1-era DB with legacy transcript rows indexed in messages_fts."""
    db = tmp_path / name
    conn = sqlite3.connect(db)
    conn.executescript(_legacy_schema())
    conn.execute(
        "INSERT INTO sessions(session_id, source, external_session_id, source_revision, "
        "parser_version, project_slug, jsonl_path, jsonl_mtime, started_at, ended_at) "
        "VALUES ('legacy-s1', 'claude', 'legacy-s1', 'legacy:1.500000', 'claude-v1', "
        "'-home-user', '/home/user.jsonl', 1.5, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z')"
    )
    conn.execute(
        "INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES "
        "('legacy-s1', 0, 'user', '2026-01-01T00:01:00Z', "
        "'legacy transcript about the postgres migration plan')"
    )
    conn.execute(
        "INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES "
        "('legacy-s1', 1, 'assistant', '2026-01-01T00:02:00Z', "
        "'the legacy migration was completed')"
    )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------- fresh DB


def test_recall_tables_created_with_expected_columns(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    assert _columns(conn, "recall_documents") == {
        "id": {"notnull": 0, "pk": 1},
        "content_sha256": {"notnull": 1, "pk": 0},
        "source_kind": {"notnull": 1, "pk": 0},
        "trust_level": {"notnull": 1, "pk": 0},
        "project_slug": {"notnull": 0, "pk": 0},
        "source_path": {"notnull": 1, "pk": 0},
        "source_updated_at": {"notnull": 1, "pk": 0},
        "indexed_at": {"notnull": 1, "pk": 0},
        "source_version": {"notnull": 1, "pk": 0},
        "text": {"notnull": 1, "pk": 0},
    }
    assert _columns(conn, "recall_events") == {
        "session_id": {"notnull": 1, "pk": 1},
        "event": {"notnull": 1, "pk": 2},
        "policy_version": {"notnull": 1, "pk": 3},
        "status": {"notnull": 1, "pk": 0},
        "attempt_count": {"notnull": 1, "pk": 0},
        "started_at": {"notnull": 0, "pk": 0},
        "finished_at": {"notnull": 0, "pk": 0},
        "selected_ids_json": {"notnull": 0, "pk": 0},
        "error_code": {"notnull": 0, "pk": 0},
    }
    assert _columns(conn, "recall_calibrations") == {
        "mode": {"notnull": 0, "pk": 1},
        "calibration_version": {"notnull": 1, "pk": 0},
        "threshold": {"notnull": 1, "pk": 0},
        "fixture_sha256": {"notnull": 1, "pk": 0},
        "created_at": {"notnull": 1, "pk": 0},
    }
    assert _columns(conn, "recall_embeddings") == {
        "document_id": {"notnull": 1, "pk": 1},
        "content_sha256": {"notnull": 1, "pk": 2},
        "adapter_fingerprint": {"notnull": 1, "pk": 3},
        "vector_json": {"notnull": 1, "pk": 0},
        "created_at": {"notnull": 1, "pk": 0},
    }
    conn.close()


def _insert_document(conn, doc_id: str = "doc-1", text: str = "postgres migration notes",
                     source_kind: str = "approved_memory", trust_level: str = "user_approved",
                     project_slug: str | None = "-home-a-api", source_path: str = "api.md") -> None:
    conn.execute(
        "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, "
        "project_slug, source_path, source_updated_at, indexed_at, source_version, text) "
        "VALUES (?, ?, ?, ?, ?, ?, '2026-08-20T10:00:00Z', '2026-08-20T10:00:01Z', 'v1', ?)",
        (doc_id, "sha-" + doc_id, source_kind, trust_level, project_slug,
         source_path, text),
    )
    conn.commit()


def test_recall_documents_check_constraints(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    for bad_kind in ("memory", "approved", "distilled", "raw"):
        with pytest.raises(sqlite3.IntegrityError):
            _insert_document(conn, doc_id="k", text="t", source_kind=bad_kind)
    for bad_trust in ("approved", "user", "distilled", "untrusted", ""):
        with pytest.raises(sqlite3.IntegrityError):
            _insert_document(conn, doc_id="t", text="t", trust_level=bad_trust)
    _insert_document(conn, doc_id="good-1")
    _insert_document(conn, doc_id="good-2", source_kind="distilled_summary",
                     trust_level="model_distilled")
    _insert_document(conn, doc_id="good-3", source_kind="raw_transcript",
                     trust_level="untrusted_transcript", project_slug=None,
                     source_path="transcript:claude:ext-1")
    assert conn.execute("SELECT COUNT(*) FROM recall_documents").fetchone()[0] == 3
    conn.close()


def test_recall_events_status_check_and_composite_primary_key(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO recall_events(session_id, event, policy_version, status, "
            "attempt_count) VALUES ('s1', 'prompt', 'recall-v1', 'done', 1)"
        )
    conn.execute(
        "INSERT INTO recall_events(session_id, event, policy_version, status, "
        "attempt_count) VALUES ('s1', 'prompt', 'recall-v1', 'running', 1)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO recall_events(session_id, event, policy_version, status, "
            "attempt_count) VALUES ('s1', 'prompt', 'recall-v1', 'running', 1)"
        )
    # Same session/event under a different policy version is a distinct row.
    conn.execute(
        "INSERT INTO recall_events(session_id, event, policy_version, status, "
        "attempt_count, started_at, finished_at, selected_ids_json, error_code) "
        "VALUES ('s1', 'prompt', 'recall-v2', 'succeeded', 2, "
        "'2026-08-20T10:00:00Z', '2026-08-20T10:00:05Z', '[\"doc-1\"]', NULL)"
    )
    assert conn.execute("SELECT COUNT(*) FROM recall_events").fetchone()[0] == 2
    conn.close()


def test_recall_calibrations_mode_primary_key(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    conn.execute(
        "INSERT INTO recall_calibrations(mode, calibration_version, threshold, "
        "fixture_sha256) VALUES ('lexical', 'cv-1', 0.1, 'fixture-hash')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO recall_calibrations(mode, calibration_version, threshold, "
            "fixture_sha256) VALUES ('lexical', 'cv-2', 0.2, 'other-hash')"
        )
    row = conn.execute(
        "SELECT calibration_version, threshold, fixture_sha256, created_at "
        "FROM recall_calibrations WHERE mode='lexical'"
    ).fetchone()
    assert row[:3] == ("cv-1", 0.1, "fixture-hash")
    assert row[3]  # created_at has a default
    conn.close()


def test_recall_embeddings_composite_primary_key(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    conn.execute(
        "INSERT INTO recall_embeddings(document_id, content_sha256, "
        "adapter_fingerprint, vector_json) VALUES ('doc-1', 'sha-a', 'fp-1', '[1,2]')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO recall_embeddings(document_id, content_sha256, "
            "adapter_fingerprint, vector_json) VALUES ('doc-1', 'sha-a', 'fp-1', '[3,4]')"
        )
    # A new adapter fingerprint (or content revision) is a distinct row.
    conn.execute(
        "INSERT INTO recall_embeddings(document_id, content_sha256, "
        "adapter_fingerprint, vector_json) VALUES ('doc-1', 'sha-a', 'fp-2', '[1,2]')"
    )
    conn.execute(
        "INSERT INTO recall_embeddings(document_id, content_sha256, "
        "adapter_fingerprint, vector_json) VALUES ('doc-1', 'sha-b', 'fp-1', '[5,6]')"
    )
    assert conn.execute("SELECT COUNT(*) FROM recall_embeddings").fetchone()[0] == 3
    conn.close()


# ------------------------------------------------------------------- FTS5


def test_recall_documents_fts_is_external_content_with_diacritic_free_tokenizer(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='recall_documents_fts'"
    ).fetchone()[0]
    assert "content='recall_documents'" in sql
    assert "content_rowid='rowid'" in sql
    assert "unicode61 remove_diacritics 2" in sql
    assert "porter" not in sql
    conn.close()


def test_recall_fts_finds_text_across_diacritics(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    _insert_document(conn, doc_id="doc-fr", text="Résumé du café à naître")
    hits = conn.execute(
        "SELECT id FROM recall_documents_fts WHERE recall_documents_fts MATCH 'resume cafe'"
    ).fetchall()
    assert [row[0] for row in hits] == ["doc-fr"]
    # Query terms with diacritics resolve to the same tokens.
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'naître'"
    ).fetchone()[0] == 1
    conn.close()


def test_recall_fts_insert_and_delete_triggers(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    _insert_document(conn, doc_id="doc-a", text="alpha transcript words")
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'alpha'"
    ).fetchone()[0] == 1
    _insert_document(conn, doc_id="doc-b", text="beta transcript words")
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'transcript'"
    ).fetchone()[0] == 2
    conn.execute("DELETE FROM recall_documents WHERE id='doc-a'")
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'alpha'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'transcript'"
    ).fetchone()[0] == 1
    conn.close()


def test_rebuild_recall_fts_repairs_index(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    _insert_document(conn, doc_id="doc-a", text="alpha beta gamma")
    _insert_document(conn, doc_id="doc-b", text="delta epsilon zeta")
    # Simulate a corrupted/emptied FTS shadow index (external content stays in the table).
    conn.execute("DELETE FROM recall_documents_fts")
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'gamma'"
    ).fetchone()[0] == 0
    rebuild_recall_fts(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'gamma'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'zeta'"
    ).fetchone()[0] == 1
    # Rebuild is idempotent: no duplicates appear.
    rebuild_recall_fts(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'gamma'"
    ).fetchone()[0] == 1
    conn.close()


def test_recall_fts_is_separate_from_messages_fts(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    conn.execute(
        "INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime) "
        "VALUES ('s-recall', '-home-a', '/home/a.jsonl', 1.0)"
    )
    conn.execute(
        "INSERT INTO messages(session_id, seq, role, text) "
        "VALUES ('s-recall', 0, 'user', 'zebra only lives in the transcript store')"
    )
    _insert_document(conn, doc_id="doc-z", text="quokka only lives in the recall store")
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'quokka'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts "
        "WHERE recall_documents_fts MATCH 'quokka'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'zebra'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts WHERE recall_documents_fts MATCH 'zebra'"
    ).fetchone()[0] == 0
    conn.close()


# ------------------------------------------------------------ migration


def test_fresh_db_records_migration_markers_1_and_2(tmp_path):
    conn = open_db(tmp_path / "dream.db")
    versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    assert versions == {1, 2}
    conn.close()


def test_migration_preserves_legacy_transcript_rows_and_messages_fts(tmp_path):
    db = _make_legacy_db(tmp_path)
    old_conn = sqlite3.connect(db)
    fts_sql_before = old_conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='messages_fts'"
    ).fetchone()[0]
    legacy_fits = old_conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'postgres'"
    ).fetchone()[0]
    old_conn.close()
    assert legacy_fits == 1

    conn = open_db(db)
    # Legacy transcript rows survive verbatim.
    rows = conn.execute(
        "SELECT seq, role, text FROM messages WHERE session_id='legacy-s1' ORDER BY seq"
    ).fetchall()
    assert rows == [
        (0, "user", "legacy transcript about the postgres migration plan"),
        (1, "assistant", "the legacy migration was completed"),
    ]
    assert conn.execute(
        "SELECT session_id, source, project_slug FROM sessions WHERE session_id='legacy-s1'"
    ).fetchone() == ("legacy-s1", "claude", "-home-user")
    # messages_fts was not recreated: identical definition and still serving
    # the legacy entries.
    fts_sql_after = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='messages_fts'"
    ).fetchone()[0]
    assert fts_sql_after == fts_sql_before
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'postgres'"
    ).fetchone()[0] == 1
    # Recall objects were added, all empty.
    for table in ("recall_documents", "recall_events", "recall_calibrations",
                  "recall_embeddings", "recall_documents_fts"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    assert versions == {1, 2}
    conn.close()

    # Reopening is idempotent and changes nothing.
    conn = open_db(db)
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'postgres'"
    ).fetchone()[0] == 1
    assert {row[0] for row in conn.execute("SELECT version FROM schema_migrations")} == {1, 2}
    conn.close()


def test_migration_creates_recall_fts_on_legacy_db_without_touching_messages(tmp_path):
    db = _make_legacy_db(tmp_path)
    conn = open_db(db)
    triggers = {row[1] for row in conn.execute(
        "SELECT type, name FROM sqlite_master WHERE type='trigger' AND name LIKE 'recall%'"
    )}
    assert triggers == {"recall_documents_ai", "recall_documents_ad"}
    _insert_document(conn, doc_id="doc-legacy", text="recall text after migration")
    assert conn.execute(
        "SELECT COUNT(*) FROM recall_documents_fts "
        "WHERE recall_documents_fts MATCH 'migration'"
    ).fetchone()[0] == 1
    # The legacy transcript index is untouched by recall writes: its two
    # 'migration' entries are still there, and the recall-only word does not
    # leak into it.
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'migration'"
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'recall'"
    ).fetchone()[0] == 0
    conn.close()
