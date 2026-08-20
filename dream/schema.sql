-- dream local memory store
--
-- One DB per host at ~/.claude/dream.db. Holds:
--   * sessions      one row per JSONL transcript file
--   * messages      filtered user/assistant turns (no thinking/hooks/tool_use noise)
--   * distilled     current per-session structured notes
--   * suggestions   proposed consolidated memory entries awaiting user review
--   * processed     incremental cursor — what's already distilled/consolidated
--   * archived_memories  retired memory files, kept for lookup but out of MEMORY.md

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    source          TEXT NOT NULL DEFAULT 'claude',
    external_session_id TEXT,
    source_revision TEXT,
    parser_version  TEXT,
    project_slug    TEXT NOT NULL,           -- cwd with '/' → '-', e.g. "-home-alice", "-home-alice-Dev-api"
    jsonl_path      TEXT NOT NULL,
    jsonl_mtime     REAL NOT NULL,           -- so we re-ingest if file grew
    started_at      TEXT,
    ended_at        TEXT,
    cwd             TEXT,
    git_branch      TEXT,
    user_msg_count  INTEGER DEFAULT 0,
    asst_msg_count  INTEGER DEFAULT 0,
    total_chars     INTEGER DEFAULT 0,
    ingested_at     TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_slug);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

CREATE TABLE IF NOT EXISTS messages (
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,           -- order within session
    role         TEXT NOT NULL,              -- 'user' | 'assistant'
    timestamp    TEXT,
    text         TEXT NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    role UNINDEXED,
    session_id UNINDEXED,
    timestamp UNINDEXED,
    content='messages',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text, role, session_id, timestamp)
    VALUES (new.rowid, new.text, new.role, new.session_id, new.timestamp);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text, role, session_id, timestamp)
    VALUES('delete', old.rowid, old.text, old.role, old.session_id, old.timestamp);
END;

CREATE TABLE IF NOT EXISTS distilled (
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
    notes_json    TEXT NOT NULL,             -- structured JSON: {facts, preferences, projects, references, references_external}
    summary       TEXT,                       -- short prose summary for ad-hoc browsing
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS suggestions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    kind          TEXT NOT NULL,             -- 'new' | 'update' | 'remove' | 'index'
    target_path   TEXT NOT NULL,             -- relative under ~/.claude/memory or project memory
    body          TEXT NOT NULL,             -- proposed file content (full, not diff — diff computed at review time)
    rationale     TEXT,                       -- why this change
    source_sessions TEXT,                     -- comma-separated session ids
    status        TEXT DEFAULT 'pending',    -- 'pending' | 'accepted' | 'rejected'
    reviewed_at   TEXT,
    sug_file      TEXT,                       -- filename of the preview .md under .suggestions/; review deletes it on resolve
    base_sha256   TEXT,
    target_existed INTEGER,
    consolidation_run_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status);

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version      INTEGER PRIMARY KEY,
    applied_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS consolidation_runs (
    run_id        TEXT PRIMARY KEY,
    provider      TEXT,
    model         TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS consolidated_distillations (
    distillation_key TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    consolidated_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES consolidation_runs(run_id) ON DELETE CASCADE
);

-- Retired memories. Files pulled out of the live memory store (and out of
-- MEMORY.md, which is loaded into every session) but worth keeping readable.
-- Not part of the distil/consolidate flow — lookup only:
--   sqlite3 ~/.claude/dream.db "SELECT body FROM archived_memories WHERE name='...'"
CREATE TABLE IF NOT EXISTS archived_memories (
    name         TEXT PRIMARY KEY,      -- former filename without .md
    title        TEXT,                  -- title as it read in the MEMORY.md index
    kind         TEXT,                  -- user | feedback | project | reference
    body         TEXT NOT NULL,         -- full original file, frontmatter included
    reason       TEXT,                  -- why it was retired
    archived_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_archived_kind ON archived_memories(kind);

-- Automatic memory recall (post-v0.0.1). Canonical document store beside the
-- transcript-only messages_fts: approved memories, distilled summaries and
-- (when opted in) raw transcripts live here, never in messages/messages_fts.
CREATE TABLE IF NOT EXISTS recall_documents (
    id                TEXT PRIMARY KEY,  -- stable UUID source identity
    content_sha256    TEXT NOT NULL,     -- SHA-256 of canonical source content
    source_kind       TEXT NOT NULL CHECK (source_kind IN ('approved_memory', 'distilled_summary', 'raw_transcript')),
    trust_level       TEXT NOT NULL CHECK (trust_level IN ('user_approved', 'model_distilled', 'untrusted_transcript')),
    project_slug      TEXT,              -- nullable normalized project identity
    source_path       TEXT NOT NULL,     -- relative or synthetic stable locator
    source_updated_at TEXT NOT NULL,     -- UTC source timestamp
    indexed_at        TEXT NOT NULL,     -- UTC index timestamp
    source_version    TEXT NOT NULL,     -- ingest / distillation / file-revision identifier
    text              TEXT NOT NULL      -- canonical searchable content (pre-render redaction)
);

CREATE TABLE IF NOT EXISTS recall_events (
    session_id        TEXT NOT NULL,
    event             TEXT NOT NULL,     -- e.g. 'session-start:startup', 'prompt'
    policy_version    TEXT NOT NULL,     -- e.g. 'recall-v1'
    status            TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    attempt_count     INTEGER NOT NULL DEFAULT 1,
    started_at        TEXT,
    finished_at       TEXT,
    selected_ids_json TEXT,
    error_code        TEXT,
    PRIMARY KEY (session_id, event, policy_version)
);

CREATE TABLE IF NOT EXISTS recall_calibrations (
    mode                TEXT PRIMARY KEY,  -- 'lexical' | 'lexical_plus_embedder' | 'lexical_plus_reranker' | 'combined'
    calibration_version TEXT NOT NULL,
    threshold           REAL NOT NULL,
    fixture_sha256      TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS recall_embeddings (
    document_id         TEXT NOT NULL,
    content_sha256      TEXT NOT NULL,
    adapter_fingerprint TEXT NOT NULL,
    vector_json         TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, content_sha256, adapter_fingerprint)
);

-- External-content FTS5 index over recall_documents (rebuildable from the
-- table via rebuild_recall_fts(); never recreates messages_fts).
CREATE VIRTUAL TABLE IF NOT EXISTS recall_documents_fts USING fts5(
    text,
    id UNINDEXED,
    content_sha256 UNINDEXED,
    source_kind UNINDEXED,
    trust_level UNINDEXED,
    project_slug UNINDEXED,
    source_path UNINDEXED,
    source_updated_at UNINDEXED,
    indexed_at UNINDEXED,
    source_version UNINDEXED,
    content='recall_documents',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS recall_documents_ai AFTER INSERT ON recall_documents BEGIN
    INSERT INTO recall_documents_fts(rowid, text, id, content_sha256, source_kind,
                                     trust_level, project_slug, source_path,
                                     source_updated_at, indexed_at, source_version)
    VALUES (new.rowid, new.text, new.id, new.content_sha256, new.source_kind,
            new.trust_level, new.project_slug, new.source_path,
            new.source_updated_at, new.indexed_at, new.source_version);
END;

CREATE TRIGGER IF NOT EXISTS recall_documents_ad AFTER DELETE ON recall_documents BEGIN
    INSERT INTO recall_documents_fts(recall_documents_fts, rowid, text, id,
                                     content_sha256, source_kind, trust_level,
                                     project_slug, source_path, source_updated_at,
                                     indexed_at, source_version)
    VALUES ('delete', old.rowid, old.text, old.id, old.content_sha256,
            old.source_kind, old.trust_level, old.project_slug, old.source_path,
            old.source_updated_at, old.indexed_at, old.source_version);
END;
