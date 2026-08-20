# Sandbox-Safe Dream Recall Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with test-first changes and a verification checkpoint after each task.

**Goal:** Make Dream's automatic recall and read-only CLI commands work in Codex's workspace sandbox without extra launch flags, while keeping the existing persistent database and host-side write pipeline.

**Architecture:** Split SQLite access into explicit read-write and immutable read-only paths. Hook context uses the immutable path and a small `/tmp` event store for first-prompt/session-start deduplication, so it never creates WAL, migration backups, or event writes beside `~/.claude/dream.db`. Ingest, distill, consolidate, and suggestion application keep using the existing read-write path and host timer.

**Tech Stack:** Python 3.11+ standard library, SQLite URI connections, atomic filesystem markers, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-automatic-memory-recall-design.md`

## Global Constraints

- Preserve `~/.claude/dream.db` and the existing memory root; do not fork state into `/tmp`.
- Read-only commands and hooks must not create or modify files under `~/.claude`.
- Hook success remains exactly one JSON object on stdout; all hook failures remain fail-open with empty stdout and exit code 0.
- Hook event markers may contain only bounded metadata and must use owner-only permissions under `/tmp`.
- Write commands retain schema migration, synchronization, WAL, backup, and memory-write behavior.
- Existing recall ranking, rendering, trust labels, scrubbing, and optional adapter contracts remain unchanged.
- Preserve all unrelated user modifications already present in the working tree.

## Task 1: Read-only SQLite boundary

**Files:**
- Modify: `dream/dream.py`
- Modify: `dream/recall_context.py`
- Modify: `dream/test_recall_context.py`
- Modify: `dream/test_recall_preflight.py`

**Interfaces:**
- Add `open_db_readonly(path: Path) -> sqlite3.Connection`.
- Keep `open_db(path: Path) -> sqlite3.Connection` as the sole write/migration opener.
- Read-only connections use `file:<absolute-path>?mode=ro&immutable=1`, `uri=True`, and `PRAGMA query_only=ON`; they never run schema migration or `PRAGMA journal_mode`.

- [x] Write a failing test proving `open_db_readonly` can query a database whose parent directory is not writable and does not create `-wal`, `-shm`, or backup files.
- [x] Write a failing test proving `dream status` and recall preflight use the read-only opener.
- [x] Run: `/home/szymon/anaconda3/bin/python -m pytest -q -p no:cacheprovider dream/test_recall_context.py dream/test_recall_preflight.py`
- [x] Implement the smallest read-only opener and route `cmd_status` plus `recall_preflight` through it.
- [x] Run the focused tests and then the full suite.

## Task 2: File-backed hook lifecycle state

**Files:**
- Create: `dream/recall_hook_state.py`
- Create: `dream/test_recall_hook_state.py`
- Modify: `dream/dream.py`

**Interfaces:**
- Add `claim_hook_event(session_id: str, event: str) -> bool`.
- Add `finish_hook_event(session_id: str, event: str, *, succeeded: bool, selected_ids: Iterable[str] = ()) -> None`.
- Add `successful_session_start_ids(session_id: str) -> frozenset[str]`.
- Store bounded JSON marker files below `/tmp/dream-recall-<uid>/`, with atomic create/replace and `0600` files / `0700` directory.

- [x] Write failing tests for first-prompt deduplication, separate session-start sources, selected-ID exclusion, stale-running retry, and owner-only marker permissions.
- [x] Run the focused state tests and confirm they fail because the module/functions are absent.
- [x] Implement the minimal file-backed state store with stale marker recovery and atomic JSON replacement.
- [x] Run the focused state tests and the full suite.

## Task 3: Read-only hook execution

**Files:**
- Modify: `dream/dream.py`
- Modify: `dream/test_recall_context.py`
- Modify: `dream/test_recall_end_to_end.py`
- Modify: `dream/recall_adapters.py`

**Interfaces:**
- `cmd_context` opens the database with `open_db_readonly`.
- Hook lifecycle calls use `recall_hook_state`, never the SQLite `recall_events` table.
- Optional embedder caching must not write to SQLite during hook recall; adapters still fall back deterministically when cache rows are absent.

- [x] Write a failing integration test that runs `dream context prompt` against a database outside the writable workspace and asserts JSON output plus no database sidecars.
- [x] Add a failing test that an enabled embedder cannot cause a read-only hook to commit.
- [x] Run the focused integration tests and confirm failure at the current SQLite write/claim boundary.
- [x] Route prompt/session-start exclusion and lifecycle bookkeeping through the file-backed state store.
- [x] Make optional adapter cache writes conditional on a writable context; hook context remains lexical when no cached vector exists.
- [x] Run focused tests and the full suite.

## Task 4: Read-only diagnostics and operational verification

**Files:**
- Modify: `dream/recall_context.py`
- Modify: `dream/test_recall_preflight.py`
- Modify: `README.md`
- Modify: `skill/SKILL.md`

- [x] Add regression coverage that `dream status`, `dream preflight`, `dream context prompt --explain`, and `dream context session-start --explain` work with read-only access to `~/.claude/dream.db`.
- [x] Document that hooks are read-only and the systemd timer is the write path; remove guidance that requires `--add-dir` for normal operation.
- [x] Run the full suite with `/home/szymon/anaconda3/bin/python -m pytest -q -p no:cacheprovider dream`.
- [x] Run live sandbox checks against the real database and assert no changes to its mtime, size, sidecars, or backup count.
- [x] Run the exact hook payloads and verify one JSON object, exit code 0, and no diagnostic text on stdout.
