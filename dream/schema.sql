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
