#!/usr/bin/env python3
"""
dream — provider-agnostic local memory consolidation.

Subcommands:
  ingest        Scan configured transcript sources into SQLite
  estimate      Show how many sessions would be distilled + subscription-limit draw
  distill       Run the configured distillation stage and store JSON notes
  consolidate   Run the configured consolidation stage and propose updates
  audit         Weekly staleness audit over the whole store (remove/patch only)
  search        FTS5 search over all ingested transcripts
  recall        Read-only approved-memory query for agents (prints the recall block)
  mcp           Read-only recall tools over MCP stdio (serve), and client install/uninstall
  review        Walk pending suggestions, accept/reject interactively
  status        Show DB stats

Safe defaults:
  * `ingest` is idempotent (content-revision checked)
  * `distill` requires --yes once you exceed --max-sessions (default 5)
  * `consolidate` writes suggestions only
  * live memory writes happen via review/curate when review.mode is auto-apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote
from typing import Any

# resolve() so the installed launcher (~/.local/bin/dream is a symlink into the repo)
# finds schema.sql / prompts next to the real file, not next to the symlink.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ingest import discover_jsonl, harvest_recall_usage, home_project_slug, ingest_file, ingest_configured_sources  # noqa: E402
from distill import distill_session, sessions_needing_distill  # noqa: E402
from consolidate import consolidate as run_consolidate  # noqa: E402
from consolidate import PatchError, patch_text, run_audit, states_deletion  # noqa: E402
from search import search as fts_search, session_summary  # noqa: E402
from backend import active_backend, generate, preflight  # noqa: E402
from config import load_config  # noqa: E402
from curate import CURATION_SCHEMA, build_curation_prompt, parse_curation_output  # noqa: E402
from recall_documents import embed_recall_documents, synchronize_recall_documents  # noqa: E402
from recall_context import append_diagnostic, configured_embedder, parse_hook_payload, recall_preflight, recent_diagnostic, run_context, hook_success_json  # noqa: E402
from recall_context import RECALL_KINDS, RecallUnavailable, agent_results, recall_for_agent, validate_agent_query, with_adapters  # noqa: E402
from recall_hook_state import claim_hook_event, finish_hook_event, successful_session_start_ids  # noqa: E402
from recall_hooks import install_hooks, uninstall_hooks  # noqa: E402
from storage import open_db_readonly  # noqa: E402
from retention_policy import POLICY_FILE, excludes_source_sessions, load_retention_policy  # noqa: E402


def _sync_recall_documents(conn: sqlite3.Connection, memory_root,
                           *, include_raw_transcripts: bool = False, constitution_path) -> bool:
    """Keep the recall store current after a successful mutation.

    Synchronization is quiet on success and non-fatal: any error (missing
    recall objects, FTS problems, ...) is reported to stderr and swallowed so
    the caller's already-committed mutation is never undone and the command
    keeps its documented behavior. Raw transcripts are opt-in. Returns
    whether synchronization succeeded.
    """
    try:
        synchronize_recall_documents(
            conn,
            memory_root,
            include_raw_transcripts=include_raw_transcripts,
            constitution_path=constitution_path,
        )
    except Exception as e:
        print(f"recall sync skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return False
    return True


def _embed_recall_documents(conn: sqlite3.Connection, config) -> None:
    """Nightly: embed new or changed recall documents after a successful sync.

    Quiet when the embedder is off. Never fails the caller: an unusable or
    unreachable embedder prints one diagnostic line to stderr and leaves the
    lexical index complete.
    """
    try:
        embedder, problem = configured_embedder(config)
        if problem:
            print(f"recall embed skipped: embedder misconfigured: {problem}", file=sys.stderr)
            return
        if embedder is None:
            return
        report = embed_recall_documents(
            conn, embedder, include_raw_transcripts=config.recall_settings().allow_raw_transcript_prompt,
        )
    except Exception as e:
        print(f"recall embed skipped: {type(e).__name__}: {e}", file=sys.stderr)
        return
    if report.error:
        print(f"recall embed skipped: {report.error} ({report.embedded} of {report.candidates} documents stored)", file=sys.stderr)
    elif report.embedded:
        print(f"Recall embeddings: stored {report.embedded} vectors.")


def _sub_label() -> str:
    """Human name of the subscription the active backend draws on."""
    return f"{active_backend()} provider"


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    _backup_before_schema_migration(conn, path)
    schema = (ROOT / "schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema)
    _migrate(conn)
    return conn


def _open_curation_readonly(path: Path) -> sqlite3.Connection:
    """Open an existing curation DB read-only, including databases with an active WAL."""
    resolved = path.expanduser().resolve()
    if not resolved.with_name(f"{resolved.name}-wal").exists():
        return open_db_readonly(resolved)
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _backup_before_schema_migration(conn: sqlite3.Connection, path: Path) -> Path | None:
    """Create one consistent SQLite backup before upgrading a legacy on-disk DB."""
    if str(path) == ":memory:" or not path.exists() or not _table_columns(conn, "sessions"):
        return None
    needs_upgrade = (
        "source" not in _table_columns(conn, "sessions")
        or "distillation_key" not in _table_columns(conn, "distilled")
        or "base_sha256" not in _table_columns(conn, "suggestions")
        or "find_text" not in _table_columns(conn, "suggestions")
        or "vector_blob" not in _table_columns(conn, "recall_embeddings")
        or not _table_columns(conn, "recall_usage")
        or not _table_columns(conn, "recall_usage_markers")
        or "path_text" not in _table_columns(conn, "recall_documents")
    )
    migrations = _table_columns(conn, "schema_migrations")
    version = conn.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0] if migrations else 0
    needs_upgrade = needs_upgrade or version < 4
    if not needs_upgrade:
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(f"{path.name}.bak-provider-migration-{stamp}")
    if not backup_path.exists():
        dest = sqlite3.connect(backup_path)
        try:
            conn.backup(dest)
        finally:
            dest.close()
        backup_path.chmod(0o600)
    return backup_path


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent additive migration from the original single-provider schema."""
    additions = {
        "sessions": {
            "source": "TEXT NOT NULL DEFAULT 'claude'",
            "external_session_id": "TEXT",
            "source_revision": "TEXT",
            "parser_version": "TEXT",
        },
        "distilled": {
            "distillation_key": "TEXT",
            "provider": "TEXT",
            "source_revision": "TEXT",
            "parser_version": "TEXT",
            "prompt_version": "TEXT",
            "provider_options": "TEXT",
            "usage_json": "TEXT",
            "duration_ms": "INTEGER",
        },
        "suggestions": {
            "sug_file": "TEXT",
            "base_sha256": "TEXT",
            "target_existed": "INTEGER",
            "consolidation_run_id": "TEXT",
            "find_text": "TEXT",
        },
        # Float32 recall vectors. No schema_migrations version: read-only
        # paths detect the column with PRAGMA table_info and fail open.
        "recall_embeddings": {
            "vector_blob": "BLOB",
            "dimension": "INTEGER",
        },
        # Pre-normalized path evidence, filled by the next sync. Read-only
        # paths detect the column and compute a NULL value on the fly.
        "recall_documents": {
            "path_text": "TEXT",
        },
    }
    with conn:
        for table, columns in additions.items():
            existing = _table_columns(conn, table)
            for name, sql_type in columns.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

        conn.execute(
            "UPDATE sessions SET external_session_id=session_id "
            "WHERE external_session_id IS NULL"
        )
        conn.execute(
            "UPDATE sessions SET source_revision='legacy:' || printf('%.6f', jsonl_mtime) "
            "WHERE source_revision IS NULL"
        )
        conn.execute(
            "UPDATE sessions SET parser_version='claude-v1' WHERE parser_version IS NULL"
        )
        conn.execute(
            "UPDATE distilled SET distillation_key='legacy:' || session_id "
            "WHERE distillation_key IS NULL"
        )
        conn.execute(
            "UPDATE distilled SET source_revision=(SELECT source_revision FROM sessions s "
            "WHERE s.session_id=distilled.session_id) WHERE source_revision IS NULL"
        )
        conn.execute(
            "UPDATE distilled SET parser_version=(SELECT parser_version FROM sessions s "
            "WHERE s.session_id=distilled.session_id) WHERE parser_version IS NULL"
        )
        conn.execute("UPDATE distilled SET prompt_version='legacy' WHERE prompt_version IS NULL")
        conn.execute(
            "UPDATE distilled SET provider=CASE WHEN model LIKE 'codex:%' THEN 'codex' ELSE 'claude' END "
            "WHERE provider IS NULL"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_source_external "
            "ON sessions(source, external_session_id) WHERE external_session_id IS NOT NULL"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_distilled_key "
            "ON distilled(distillation_key) WHERE distillation_key IS NOT NULL"
        )

        cursor = conn.execute(
            "SELECT value FROM meta WHERE key='last_consolidate_at'"
        ).fetchone()
        if cursor:
            run_id = "legacy-watermark"
            conn.execute(
                "INSERT OR IGNORE INTO consolidation_runs(run_id, provider, model) "
                "VALUES (?, 'legacy', 'legacy')",
                (run_id,),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO consolidated_distillations(distillation_key, session_id, run_id)
                SELECT d.distillation_key, d.session_id, ?
                FROM distilled d JOIN sessions s ON s.session_id=d.session_id
                WHERE COALESCE(s.ended_at, s.started_at, d.distilled_at) <= ?
                """,
                (run_id, cursor[0]),
            )
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (1)")
        # Version 2: automatic memory recall objects (recall_documents and the
        # other recall tables plus their FTS5 index and triggers are created by
        # schema.sql above; messages_fts is never dropped or recreated).
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (2)")
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (3)")
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (4)")


def rebuild_recall_fts(conn: sqlite3.Connection) -> None:
    """Rebuild recall_documents_fts from recall_documents.

    Idempotent and safe to run inside a write transaction: the FTS5 'rebuild'
    command reindexes every row of the external-content table, recovering a
    missing or corrupt index. Callers must keep it transactional so a failure
    rolls back and retains the previous complete index.
    """
    conn.execute(
        "INSERT INTO recall_documents_fts(recall_documents_fts) VALUES ('rebuild')"
    )


def _color(s: str, c: str) -> str:
    if not sys.stdout.isatty():
        return s
    codes = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36, "grey": 90, "bold": 1}
    return f"\033[{codes.get(c, 0)}m{s}\033[0m"


_INDEX_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")


def _index_target(line: str) -> str | None:
    """Return the target filename of a MEMORY.md index link line, or None if not a link line."""
    m = _INDEX_LINK_RE.search(line)
    return m.group(1) if m else None


def _merge_index(existing: str, proposed: str) -> str:
    """Append-only merge for MEMORY.md.

    Keep the existing index verbatim, then append only the proposed link lines
    whose target file isn't already indexed. This can never drop or revert an
    existing line — unlike a wholesale overwrite, which loses entries when the
    proposed body was built from a stale/truncated snapshot (see consolidate's
    MAX_CURRENT_MEMORY_CHARS cap and multiple index suggestions per run).

    Tradeoff: an already-present file's one-line hook is NOT refreshed here; the
    topic file stays the source of truth.
    """
    existing_lines = existing.splitlines()
    have = {t for line in existing_lines if (t := _index_target(line))}
    additions = [
        line for line in proposed.splitlines()
        if (t := _index_target(line)) and t not in have
    ]
    if not additions:
        return existing
    body = existing.rstrip("\n")
    return body + "\n" + "\n".join(additions) + "\n"


def _prune_orphaned_index_lines(memory_root: Path, *, constitution_path) -> bool:
    """Drop MEMORY.md link lines whose target file no longer exists on disk.

    Deterministic and LLM-independent: runs after every suggestion batch (whether or
    not it contained a `remove`) so MEMORY.md never drifts from what's actually on
    disk, regardless of whether a matching `index` suggestion was proposed, or the
    file vanished some other way (e.g. a manual `rm`). Returns whether it rewrote
    the file, so the caller can log it.
    """
    index_path = memory_root / "MEMORY.md"
    if not index_path.exists():
        return False
    from character.safety import is_memory_control_source
    if is_memory_control_source(index_path, memory_root=memory_root, constitution_path=constitution_path):
        raise ValueError("memory index aliases a protected input")
    lines = index_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [
        line for line in lines
        if (t := _index_target(line)) is None or (memory_root / t).exists()
    ]
    if kept == lines:
        return False
    if is_memory_control_source(index_path, memory_root=memory_root, constitution_path=constitution_path):
        raise ValueError("memory index aliases a protected input")
    index_path.write_text("".join(kept), encoding="utf-8")
    return True


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = open_db(args.db)
    harvested = harvest_recall_usage(conn)
    if harvested:
        print(f"recall usage: counted {harvested} hook marker{'' if harvested == 1 else 's'}.")
    if args.projects:
        seen = ingested = msgs = 0
        for path in discover_jsonl(Path(args.projects)):
            seen += 1
            changed, count = ingest_file(path, conn, force=args.force)
            if changed:
                ingested += 1
                msgs += count
        results = [("claude-legacy", seen, ingested, msgs)]
    else:
        results = ingest_configured_sources(
            conn,
            config=args.config,
            selected=set(args.source) if args.source else None,
            force=args.force,
        )
    for name, seen, ingested, messages in results:
        print(f"{name}: scanned {seen}, ingested {ingested}, {messages} messages stored.")
    print(f"DB: {args.db}")
    if _sync_recall_documents(conn, args.config.memory_root, constitution_path=args.config.data['character']['constitution_path']):
        _embed_recall_documents(conn, args.config)
    failures = getattr(results, "failures", [])
    if failures:
        print(f"Ingestion completed with {len(failures)} source/session errors. Failed items remain eligible for retry.",
              file=sys.stderr)
    return 1 if failures else 0


def cmd_estimate(args: argparse.Namespace) -> int:
    conn = open_db_readonly(args.db)
    candidates = sessions_needing_distill(
        conn, min_chars=args.min_chars, project_slug=args.project, config=args.config,
        refresh_config=getattr(args, "refresh", False), model=getattr(args, "model", None),
    )
    if not candidates:
        print("Nothing to distill.")
        return 0
    total_chars = sum(c for _, c in candidates)
    # Very rough: 4 chars ≈ 1 token; output assumed ~500 tokens per session
    in_tokens = total_chars / 4
    out_tokens = len(candidates) * 500
    print(f"Sessions to distill: {len(candidates)}")
    print(f"Total chars:         {total_chars:,}")
    print(f"Est. input tokens:   {int(in_tokens):,}")
    print(f"Est. output tokens:  {int(out_tokens):,}")
    print(f"Provider calls:      {len(candidates)} via {_sub_label()}")
    if args.verbose:
        for sid, n in candidates[:20]:
            print(f"  {sid}  {n:,} chars")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    conn = open_db(args.db)
    candidates = sessions_needing_distill(
        conn, min_chars=args.min_chars, project_slug=args.project, config=args.config,
        refresh_config=getattr(args, "refresh", False), model=getattr(args, "model", None),
    )
    if not candidates:
        print("Nothing to distill.")
        return 0

    n_total = len(candidates)
    if args.limit:
        candidates = candidates[:args.limit]

    if len(candidates) > args.max_sessions and not args.yes:
        print(_color(
            f"Would distill {len(candidates)} sessions (of {n_total} pending). "
            f"Each call uses the configured {_sub_label()}.",
            "yellow",
        ))
        print(f"Pass --yes to proceed, or --limit N to cap.")
        return 2

    total_in = total_out = total_cache_w = total_cache_r = 0
    failures = 0
    t_start = __import__("time").monotonic()
    for i, (sid, chars) in enumerate(candidates, 1):
        try:
            res = distill_session(conn, sid, model=args.model, config=args.config, memory_root=args.config.memory_root)
        except Exception as e:
            print(f"  {_color('!', 'red')} {sid[:8]}…: {type(e).__name__}: {e}")
            failures += 1
            continue
        if res is None:
            continue
        total_in += res.input_tokens or 0
        total_out += res.output_tokens or 0
        total_cache_w += res.cache_creation_tokens or 0
        total_cache_r += res.cache_read_tokens or 0
        n_findings = sum(len(v) for k, v in res.notes.items() if isinstance(v, list))
        summary = res.summary[:70] if res.summary else "(no summary)"
        elapsed = __import__("time").monotonic() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(candidates) - i) / rate if rate > 0 else 0
        token = lambda value, width: f"{value:>{width}}" if value is not None else f"{'-':>{width}}"
        print(f"  [{i:>4}/{len(candidates)}] {_color('✓', 'green')} {sid[:8]}…  {chars:>6,}ch  "
              f"in={token(res.input_tokens, 4)} out={token(res.output_tokens, 3)} "
              f"cw={token(res.cache_creation_tokens, 5)} cr={token(res.cache_read_tokens, 5)} "
              f"findings={n_findings:>2}  eta={int(eta)//60:02d}:{int(eta)%60:02d}  {summary}")
    print(f"\nTotal: in={total_in:,} out={total_out:,} cache_w={total_cache_w:,} cache_r={total_cache_r:,}")
    print(f"Provider: {_sub_label()}.")
    return 1 if failures else 0


def _subset_embedder(config):
    """The configured embedder for subset selection, or None to rank lexically.

    Building it contacts nothing; an unusable configuration prints one line.
    """
    try:
        embedder, problem = configured_embedder(config)
    except Exception as e:
        embedder, problem = None, f"{type(e).__name__}: {e}"
    if problem:
        print(f"subset selection: embedder misconfigured ({problem}); ranking lexically", file=sys.stderr)
    return embedder


def cmd_consolidate(args: argparse.Namespace) -> int:
    conn = open_db(args.db)
    memory_root = Path(args.memory)
    memory_root.mkdir(parents=True, exist_ok=True)
    print(f"Consolidating into suggestions for {memory_root}")
    mode = getattr(args, "mode", None)
    embedder = _subset_embedder(args.config) if (mode or args.config.consolidate_settings().mode) == "subset" else None
    suggestions = run_consolidate(
        conn, memory_root,
        since_iso=args.since,
        model=args.model,
        max_sessions=args.max_sessions,
        config=args.config,
        mode=mode,
        embedder=embedder,
    )
    return _print_suggestions(memory_root, suggestions)


def _print_suggestions(memory_root: Path, suggestions) -> int:
    if not suggestions:
        print("No suggestions produced.")
        return 0
    print(f"\nWrote {len(suggestions)} suggestions to {memory_root}/.suggestions/")
    for s in suggestions:
        kind_color = {"new": "green", "update": "yellow", "remove": "red", "index": "blue",
                      "patch": "magenta"}.get(s.kind, "grey")
        print(f"  [{_color(s.kind, kind_color):>14}] {s.target_path}  — {s.rationale[:80]}")
    print(f"\nRun `dream review` to walk through them.")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Weekly staleness audit; in full mode the nightly consolidation already audits."""
    if args.config.consolidate_settings().mode == "full" and not getattr(args, "force", False):
        print("consolidate.mode is full: the nightly consolidation already audits the whole store; "
              "skipping. Pass --force to audit anyway.")
        return 0
    conn = open_db(args.db)
    memory_root = Path(args.memory)
    print(f"Auditing {memory_root} for stale memories")
    suggestions = run_audit(conn, memory_root, model=args.model, config=args.config)
    return _print_suggestions(memory_root, suggestions)


def cmd_search(args: argparse.Namespace) -> int:
    conn = open_db_readonly(args.db)
    hits = fts_search(conn, args.query, limit=args.limit, role=args.role, project_slug=args.project)
    if not hits:
        print("No matches.")
        return 0
    for h in hits:
        ts = (h.timestamp or "")[:19]
        role_c = _color(h.role.upper(), "cyan" if h.role == "user" else "magenta")
        print(f"\n{role_c}  {ts}  {_color(h.project_slug, 'grey')}  {_color('('+h.session_id[:8]+')', 'grey')}")
        print(f"  {h.snippet}")
    print(f"\n{len(hits)} hits.")
    return 0


def _recall_unavailable(reason: str) -> int:
    """One bounded stderr line, nothing on stdout, exit 0: agents carry on without recall."""
    print(f"dream recall: unavailable ({' '.join(str(reason).split())[:200]})", file=sys.stderr)
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    """Agent-side recall over approved memory, read-only and fail-open.

    Without ``--kinds`` it reads approved memory and distilled summaries.
    Usage errors (an empty query, ``raw_transcript`` without the opt-in)
    exit 2. A missing or busy database, an unmigrated schema or disabled
    recall print one ``unavailable`` line to stderr and exit 0. The command
    writes nothing: no markers, cache rows, embeddings or diagnostics.
    """
    settings = args.config.recall_settings()
    try:
        query_text, kinds = validate_agent_query(args.query, args.kinds, settings)
    except ValueError as exc:
        print(f"dream recall: {exc}", file=sys.stderr)
        return 2
    conn = None
    try:
        if not args.db.exists():
            return _recall_unavailable("database does not exist")
        conn = open_db_readonly(args.db)
        # Building the embedder contacts nothing; an unusable one means lexical recall.
        embedder, embedder_problem = configured_embedder(args.config)
        if embedder_problem:
            settings = with_adapters(settings, embedder=False)
        try:
            cwd = os.getcwd()
        except OSError:
            cwd = None
        result = recall_for_agent(
            conn, settings, query_text=query_text, cwd=cwd, memory_root=args.config.memory_root,
            constitution_path=args.config.data["character"]["constitution_path"], project_slug=args.project,
            kinds=kinds, limit=args.limit, budget_codepoints=settings.prompt_budget_codepoints, embedder=embedder,
        )
    except RecallUnavailable as exc:
        return _recall_unavailable(str(exc))
    except sqlite3.Error as exc:
        return _recall_unavailable(f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        # Other messages can carry absolute paths (an unreadable policy file); the type is enough.
        return _recall_unavailable(type(exc).__name__)
    finally:
        if conn is not None:
            conn.close()
    try:
        if args.json:
            fallback_reason = result.diagnostics.fallback_reason or ("embedder-misconfigured" if embedder_problem else None)
            print(json.dumps({
                "mode": result.mode,
                "calibration_version": result.calibration_version,
                "fallback_reason": fallback_reason,
                "results": agent_results(result),
                "context": result.rendered_context,
            }, ensure_ascii=False))
        elif result.rendered_context:
            print(result.rendered_context)
    except (UnicodeError, OSError) as exc:
        # A stdout that cannot encode the marker (latin-1 locale) or is closed: no traceback.
        return _recall_unavailable(type(exc).__name__)
    return 0


def _backup_memory(memory_root: Path) -> Path:
    """Snapshot every top-level *.md (the flat one-file-per-memory tree + MEMORY.md)
    before an unattended apply, so a wrong auto-accepted suggestion is one `cp` to undo.
    Kept outside memory_root (consolidate globs *.md there) and pruned to the last 14."""
    import shutil, datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = memory_root.parent / "memory-backups"
    dest = root / ts
    dest.mkdir(parents=True, exist_ok=True)
    for f in memory_root.glob("*.md"):
        shutil.copy2(f, dest / f.name)
    keep = sorted(p for p in root.iterdir() if p.is_dir())
    for old in keep[:-14]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def _resolved_body(kind: str, old: str, body: str | None) -> str:
    """What review actually writes: index append-merges, everything else replaces wholesale."""
    return _merge_index(old, body or "") if kind == "index" else (body or "")


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def apply_patch(target: Path, find: str, replace: str, *, expected_sha: str | None,
                allow_empty: bool = False) -> str:
    """Write one accepted `patch` suggestion; the only writer for patches.

    Refuses (PatchError, nothing written) unless the target exists with the
    expected SHA-256, is NFC, and contains the NFC-normalized `find` exactly
    once. An empty replacement needs ``allow_empty`` (a stated deletion).
    Callers run the containment, protected-file and retention checks and take
    the backup first, exactly as for `update`. Returns the written text.
    """
    target = Path(target)
    if not expected_sha:
        raise PatchError("patch requires a base SHA precondition")
    if not target.exists():
        raise PatchError(f"patch target does not exist: {target.name}")
    data = target.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise PatchError("target changed since the suggestion was made")
    patched = patch_text(data.decode("utf-8"), find, replace, allow_empty=allow_empty)
    target.write_text(patched, encoding="utf-8", newline="")
    return patched


def _patch_fields(conn: sqlite3.Connection, sug_id) -> tuple[str | None, str]:
    """Return (find_text, rationale) of a suggestion; (None, "") on a pre-patch schema."""
    columns = _table_columns(conn, "suggestions")
    if "find_text" not in columns:
        return None, ""
    rationale = "rationale" if "rationale" in columns else "''"
    row = conn.execute(f"SELECT find_text, {rationale} FROM suggestions WHERE id=?", (sug_id,)).fetchone()
    return (row[0], row[1] or "") if row else (None, "")


def _find_text(conn: sqlite3.Connection, sug_id) -> str | None:
    return _patch_fields(conn, sug_id)[0]


def _apply_suggestion(conn, memory_root: Path, sug_dir: Path,
                      sug_id, kind, target_path, new_content, sug_file,
                      sync_recall: bool = True, *, constitution_path) -> bool:
    """Accept one suggestion: write/merge/delete the target, mark accepted, drop the preview.

    Refuses (marks 'rejected' instead of touching the filesystem) if target_path would
    resolve outside memory_root, or if a 'remove' suggestion targets MEMORY.md itself
    (the index file, never deletable via this path). target_path comes from LLM output
    and is otherwise unvalidated before hitting unlink()/write_text().
    """
    tgt = memory_root / target_path
    resolved_root = memory_root.resolve()
    resolved_target = tgt.resolve()
    contained = resolved_target.is_relative_to(resolved_root)
    from character.safety import is_memory_control_source
    # MEMORY.md is never deleted, and never patched: index lines are only appended.
    protected = kind in ("remove", "patch") and resolved_target == resolved_root / "MEMORY.md"
    if is_memory_control_source(tgt, memory_root=memory_root, constitution_path=constitution_path):
        _reject_suggestion(conn, sug_dir, sug_id, sug_file)
        print(f"  rejected #{sug_id}: the constitution is human-controlled")
        return False
    if resolved_target == (resolved_root / POLICY_FILE).resolve():
        _reject_suggestion(conn, sug_dir, sug_id, sug_file)
        print(f"  rejected #{sug_id}: the local retention policy is user-controlled")
        return False
    state = conn.execute(
        "SELECT base_sha256, target_existed, source_sessions FROM suggestions WHERE id=?", (sug_id,)
    ).fetchone()
    policy = load_retention_policy(memory_root)
    if kind != "remove" and (
        policy.excludes(source_path=target_path, text=new_content or "")
        or policy.excludes_memory_path(memory_root, target_path)
        or excludes_source_sessions(conn, policy, state[2] if state else None)
    ):
        _reject_suggestion(conn, sug_dir, sug_id, sug_file)
        print(f"  rejected #{sug_id}: excluded by local memory retention policy")
        return False
    base_sha = state[0] if state else None
    target_existed = state[1] if state else None
    conflict = (
        base_sha is not None and _file_sha256(tgt) != base_sha
    ) or (
        base_sha is None and target_existed == 0 and tgt.exists()
    )
    if conflict:
        print(f"  conflict #{sug_id} ({target_path}): target changed since suggestion")
        return False
    if kind == "patch" and contained and not protected:
        find, rationale = _patch_fields(conn, sug_id)
        allow_empty = states_deletion(rationale)
        try:
            patched = patch_text(tgt.read_bytes().decode("utf-8") if tgt.exists() else "",
                                 find or "", new_content or "", allow_empty=allow_empty)
        except (PatchError, OSError, UnicodeDecodeError) as exc:
            print(f"  deferred #{sug_id}: {exc}")
            return False
        if policy.excludes(source_path=target_path, text=patched):
            _reject_suggestion(conn, sug_dir, sug_id, sug_file)
            print(f"  rejected #{sug_id}: excluded by local memory retention policy")
            return False
    if contained and not protected:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if kind == "remove" and tgt.exists():
            tgt.unlink()
        elif kind in ("new", "update", "index"):
            tgt.write_text(new_content, encoding="utf-8")
        elif kind == "patch":
            try:
                apply_patch(tgt, find or "", new_content or "", expected_sha=base_sha, allow_empty=allow_empty)
            except PatchError as exc:
                print(f"  deferred #{sug_id}: {exc}")
                return False
        status = "accepted"
    else:
        print(f"  refusing #{sug_id} ({target_path}): unsafe target, marking rejected")
        status = "rejected"
    conn.execute("UPDATE suggestions SET status=?, reviewed_at=CURRENT_TIMESTAMP WHERE id=?", (status, sug_id))
    conn.commit()
    if sug_file:
        f = sug_dir / sug_file
        if f.exists():
            f.unlink()
    if sync_recall and status == "accepted":
        _sync_recall_documents(conn, memory_root, constitution_path=constitution_path)
    return status == "accepted"


def _pending_suggestions(conn: sqlite3.Connection) -> list[sqlite3.Row | tuple]:
    columns = _table_columns(conn, "suggestions")
    required = {"id", "kind", "target_path", "body", "status"}
    if not required.issubset(columns):
        return []
    selected = []
    for name in (
        "id", "kind", "target_path", "body", "rationale", "source_sessions",
        "sug_file", "base_sha256", "target_existed",
    ):
        selected.append(name if name in columns else f"NULL AS {name}")
    return conn.execute(
        f"SELECT {', '.join(selected)} FROM suggestions "
        "WHERE status='pending' ORDER BY id"
    ).fetchall()


def _safe_suggestion_target(memory_root: Path, target_path: str) -> Path | None:
    """Resolve a suggestion target only when it remains inside memory_root."""
    root = memory_root.resolve()
    target = (memory_root / target_path).resolve()
    return target if target.is_relative_to(root) else None


def _curation_target_snapshot(memory_root: Path, row: tuple, find_text: str | None = None) -> dict[str, Any]:
    """Build the model-facing snapshot without reading an unsafe target path."""
    sug_id, kind, target_path, body, rationale, sources, sug_file, base_sha, existed = row
    target = _safe_suggestion_target(memory_root, target_path)
    if target is None:
        return {
            "id": sug_id,
            "kind": kind,
            "target_path": target_path,
            "body": body,
            "rationale": rationale,
            "source_sessions": sources,
            "sug_file": sug_file,
            "base_sha256": base_sha,
            "target_existed": bool(existed),
            "current_body": "",
            "current_sha256": None,
            "conflict": True,
            "find_text": find_text,
        }

    target_existed = target.exists()
    current_body = target.read_text(encoding="utf-8") if target_existed else ""
    current_sha = _file_sha256(target) if target_existed else None
    conflict = (
        base_sha is not None and current_sha != base_sha
    ) or (
        base_sha is None and existed == 0 and target_existed
    )
    return {
        "id": sug_id,
        "kind": kind,
        "target_path": target_path,
        "body": body,
        "rationale": rationale,
        "source_sessions": sources,
        "sug_file": sug_file,
        "base_sha256": base_sha,
        "target_existed": target_existed,
        "current_body": current_body,
        "current_sha256": current_sha,
        "conflict": conflict,
        "find_text": find_text,
    }


def _drop_suggestion_preview(sug_dir: Path, sug_file: str | None) -> None:
    if sug_file:
        (sug_dir / sug_file).unlink(missing_ok=True)


def _reject_suggestion(conn: sqlite3.Connection, sug_dir: Path, sug_id, sug_file) -> None:
    with conn:
        conn.execute(
            "UPDATE suggestions SET status='rejected', reviewed_at=CURRENT_TIMESTAMP WHERE id=?",
            (sug_id,),
        )
    _drop_suggestion_preview(sug_dir, sug_file)


def _curation_recheck(memory_root: Path, snapshot: dict[str, Any]) -> tuple[Path | None, str, str | None, bool]:
    """Reread a target and return path, body, SHA, and existence for the apply gate."""
    target = _safe_suggestion_target(memory_root, snapshot["target_path"])
    if target is None:
        return None, "", None, False
    exists = target.exists()
    body = target.read_text(encoding="utf-8") if exists else ""
    sha = _file_sha256(target) if exists else None
    return target, body, sha, exists


def _apply_curated_batch(conn, memory_root: Path, snapshots, decisions, *, constitution_path) -> int:
    by_id = {snapshot["id"]: snapshot for snapshot in snapshots}
    expected_by_target: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        target = _safe_suggestion_target(memory_root, snapshot["target_path"])
        if target is not None:
            expected_by_target.setdefault(
                str(target),
                {
                    "body": snapshot["current_body"],
                    "sha256": snapshot["current_sha256"],
                    "existed": snapshot["target_existed"],
                    "host_applied": False,
                },
            )
    applied: list[int] = []
    rejected: list[int] = []
    deferred: list[int] = []
    conflicts: list[int] = []
    unsafe: list[int] = []
    backup: Path | None = None

    for sug_id in sorted(decisions):
        snapshot = by_id[sug_id]
        decision = decisions[sug_id]
        kind = snapshot["kind"]
        target_path = snapshot["target_path"]
        sug_file = snapshot["sug_file"]
        if decision.decision == "reject":
            _reject_suggestion(conn, memory_root / ".suggestions", sug_id, sug_file)
            rejected.append(sug_id)
            continue
        if decision.decision == "defer":
            deferred.append(sug_id)
            continue
        if decision.decision == "merge" and kind == "remove":
            deferred.append(sug_id)
            print(f"deferred #{sug_id} ({target_path}): merge is unsupported for remove suggestions")
            continue

        target = _safe_suggestion_target(memory_root, target_path)
        if target is None:
            _reject_suggestion(conn, memory_root / ".suggestions", sug_id, sug_file)
            rejected.append(sug_id)
            unsafe.append(sug_id)
            print(f"conflict #{sug_id}: unsafe target path; rejected by host safety check")
            continue
        if kind in ("remove", "patch") and target == memory_root.resolve() / "MEMORY.md":
            _reject_suggestion(conn, memory_root / ".suggestions", sug_id, sug_file)
            rejected.append(sug_id)
            unsafe.append(sug_id)
            print(f"unsafe #{sug_id} ({target_path}): protected target; rejected by host safety check")
            continue

        rechecked_target, old, current_sha, target_existed = _curation_recheck(memory_root, snapshot)
        expected = expected_by_target[str(target)]
        if (
            rechecked_target is None
            or old != expected["body"]
            or current_sha != expected["sha256"]
            or target_existed != expected["existed"]
        ):
            deferred.append(sug_id)
            conflicts.append(sug_id)
            print(f"conflict #{sug_id} ({target_path}): target changed during curation")
            continue

        if (
            not expected["host_applied"]
            and snapshot["conflict"]
            and decision.decision == "accept"
        ):
            deferred.append(sug_id)
            conflicts.append(sug_id)
            print(f"conflict #{sug_id} ({target_path}): initial conflict; accept deferred")
            continue
        proposed_body = snapshot["body"] if decision.decision == "accept" else decision.body
        # A merge supplies a complete body, so a merged patch applies as an update.
        apply_kind = "update" if kind == "patch" and decision.decision == "merge" else kind
        new_content = _resolved_body(apply_kind, old, proposed_body)
        if backup is None:
            backup = _backup_memory(memory_root)
        with conn:
            conn.execute(
                "UPDATE suggestions SET base_sha256=?, target_existed=? WHERE id=?",
                (expected["sha256"], int(expected["existed"]), sug_id),
            )
            if apply_kind != kind:
                # Store the full merged body too: a pending update must never hold a patch span.
                conn.execute(
                    "UPDATE suggestions SET kind='update', find_text=NULL, body=? WHERE id=?",
                    (proposed_body, sug_id),
                )
        try:
            applied_ok = _apply_suggestion(
                conn,
                memory_root,
                memory_root / ".suggestions",
                sug_id,
                apply_kind,
                target_path,
                new_content,
                sug_file,
                constitution_path=constitution_path,
        )
        except Exception as exc:
            print(
                f"unsafe application #{sug_id} ({target_path}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            unsafe.append(sug_id)
            continue
        if applied_ok:
            applied.append(sug_id)
            _, expected["body"], expected["sha256"], expected["existed"] = (
                _curation_recheck(memory_root, snapshot)
            )
            expected["host_applied"] = True
        elif sug_id not in conflicts:
            deferred.append(sug_id)
            conflicts.append(sug_id)

    if applied:
        if backup is None:
            backup = _backup_memory(memory_root)
        if _prune_orphaned_index_lines(memory_root, constitution_path=constitution_path):
            _sync_recall_documents(conn, memory_root, constitution_path=constitution_path)
            print("pruned orphaned MEMORY.md links")
    print(f"applied IDs: {applied}")
    print(f"rejected IDs: {rejected}")
    print(f"deferred IDs: {deferred}")
    print(f"conflict IDs: {conflicts}")
    print(f"backup: {backup if backup is not None else 'none'}")
    if unsafe:
        print(f"unsafe IDs: {unsafe}", file=sys.stderr)
    return 1 if unsafe else 0


def _cmd_curate_configured(args: argparse.Namespace, conn, memory_root: Path, rows) -> int:
    """Curate one complete pending batch through the configured review provider."""
    if args.config.data["review"]["mode"] != "auto-apply":
        conn.close()
        print("review.mode is suggest-only; curation disabled, leaving suggestions pending")
        return 0
    if not rows:
        conn.close()
        print("No pending suggestions.")
        return 0

    try:
        snapshots = [
            _curation_target_snapshot(memory_root, row, _find_text(conn, row[0]) if row[1] == "patch" else None)
            for row in rows
        ]
        prompt = build_curation_prompt(snapshots, memory_root)
        result = generate("review", prompt, CURATION_SCHEMA, config=args.config)
        decisions = parse_curation_output(result.output, {row[0] for row in rows})
        missing_merge_bodies = [
            sug_id for sug_id, decision in decisions.items()
            if decision.decision == "merge" and not isinstance(decision.body, str)
        ]
        if missing_merge_bodies:
            raise ValueError(
                "merge decisions require string bodies: "
                + ", ".join(str(sug_id) for sug_id in sorted(missing_merge_bodies))
            )
    except Exception as exc:
        conn.close()
        print(f"curation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    provider = getattr(result, "provider", None) or "unknown"
    model = getattr(result, "model", None) or "(provider default)"
    print(f"provider: {provider} / {model}")
    dry_run = bool(getattr(args, "dry_run", False))
    for snapshot in snapshots:
        decision = decisions[snapshot["id"]]
        suffix = f" body_chars={len(decision.body)}" if decision.body is not None else ""
        print(
            f"decision #{snapshot['id']}: {decision.decision} — {decision.reason}{suffix}"
        )
    conn.close()
    if dry_run:
        print("dry-run: no files, previews, statuses, or backups changed")
        return 0
    write_conn = open_db(args.db)
    try:
        return _apply_curated_batch(write_conn, memory_root, snapshots, decisions, constitution_path=args.config.data['character']['constitution_path'])
    finally:
        write_conn.close()


def cmd_suggestions(args: argparse.Namespace) -> int:
    memory_root = Path(args.memory)
    sug_dir = memory_root / ".suggestions"

    if args.suggestions_cmd == "curate-configured":
        if not Path(args.db).exists():
            print("No pending suggestions.")
            return 0
        conn = None
        try:
            conn = _open_curation_readonly(args.db)
            rows = _pending_suggestions(conn)
        except Exception as exc:
            if conn is not None:
                conn.close()
            print(f"curation error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return _cmd_curate_configured(args, conn, memory_root, rows)

    conn = open_db_readonly(args.db) if args.suggestions_cmd == "list" else open_db(args.db)
    rows = _pending_suggestions(conn)

    if args.suggestions_cmd == "list":
        items = []
        for row in rows:
            sug_id, kind, target, body, rationale, sources, preview, base_sha, existed = row
            path = memory_root / target
            conflict = (
                base_sha is not None and _file_sha256(path) != base_sha
            ) or (base_sha is None and existed == 0 and path.exists())
            items.append({
                "id": sug_id,
                "kind": kind,
                "target_path": target,
                "body": body,
                "rationale": rationale,
                "source_sessions": [s for s in (sources or "").split(",") if s],
                "preview_file": preview,
                "base_sha256": base_sha,
                "target_existed": bool(existed),
                "current_sha256": _file_sha256(path),
                "conflict": conflict,
            })
            if kind == "patch":
                items[-1]["find"] = _find_text(conn, sug_id)
        print(json.dumps({"suggestions": items}, ensure_ascii=False, indent=2))
        return 0

    if args.suggestions_cmd in {"apply-all", "apply-configured"}:
        if args.config.data["review"]["mode"] != "auto-apply":
            if args.suggestions_cmd == "apply-configured":
                print("review.mode is suggest-only; leaving suggestions pending")
                return 0
            print("review.mode is suggest-only; refusing apply-all", file=sys.stderr)
            return 2
        if not rows:
            print("No pending suggestions.")
            return 0
        backup = _backup_memory(memory_root)
        print(f"Backup: {backup}")
        conflicts = 0
        for row in rows:
            sug_id, kind, target, body, _rationale, _sources, preview, _base, _existed = row
            path = memory_root / target
            old = path.read_text(encoding="utf-8") if path.exists() else ""
            new_content = _resolved_body(kind, old, body)
            if not _apply_suggestion(conn, memory_root, sug_dir, sug_id, kind, target, new_content, preview, constitution_path=args.config.data['character']['constitution_path']):
                conflicts += 1
        if _prune_orphaned_index_lines(memory_root, constitution_path=args.config.data['character']['constitution_path']):
            print("pruned orphaned MEMORY.md links")
        return 3 if conflicts else 0

    row = next((row for row in rows if row[0] == args.id), None)
    if row is None:
        print(f"pending suggestion #{args.id} not found", file=sys.stderr)
        return 2
    sug_id, kind, target, body, _rationale, _sources, preview, _base, _existed = row

    if args.suggestions_cmd == "reject":
        with conn:
            conn.execute(
                "UPDATE suggestions SET status='rejected', reviewed_at=CURRENT_TIMESTAMP WHERE id=?",
                (sug_id,),
            )
        if preview:
            (sug_dir / preview).unlink(missing_ok=True)
        return 0

    path = memory_root / target
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if args.suggestions_cmd == "merge":
        body = Path(args.body_file).read_text(encoding="utf-8")
        with conn:
            conn.execute(
                "UPDATE suggestions SET body=?, base_sha256=?, target_existed=? WHERE id=?",
                (body, _file_sha256(path), int(path.exists()), sug_id),
            )
            if kind == "patch":
                # The merge body is the complete file, so the patch becomes an update.
                conn.execute("UPDATE suggestions SET kind='update', find_text=NULL WHERE id=?", (sug_id,))
        kind = "update" if kind == "patch" else kind
    new_content = _resolved_body(kind, old, body)
    backup = _backup_memory(memory_root)
    print(f"Backup: {backup}")
    return 0 if _apply_suggestion(
        conn, memory_root, sug_dir, sug_id, kind, target, new_content, preview
    , constitution_path=args.config.data['character']['constitution_path']) else 3


def cmd_review(args: argparse.Namespace) -> int:
    """Walk pending suggestions; for each, show diff vs target and prompt accept/reject/skip.

    With --yes, accept every pending suggestion non-interactively (the nightly path):
    snapshot memory first, then apply each and log it. --dry-run shows the plan without writing.
    """
    import difflib
    conn = open_db(args.db)
    rows = [row[:7] for row in _pending_suggestions(conn)]
    if not rows:
        print("No pending suggestions.")
        return 0

    memory_root = Path(args.memory)
    sug_dir = memory_root / ".suggestions"

    auto = getattr(args, "yes", False)
    dry = getattr(args, "dry_run", False)
    if auto:
        if dry:
            print(f"{len(rows)} pending — dry-run, nothing will be written.")
        else:
            backup = _backup_memory(memory_root)
            print(f"{len(rows)} pending. Auto-applying into {memory_root}; backup: {backup}")
        for sug_id, kind, target_path, body, rationale, sources, sug_file in rows:
            tgt = memory_root / target_path
            old = tgt.read_text(encoding="utf-8") if tgt.exists() else ""
            new_content = _resolved_body(kind, old, body)
            if kind == "patch":
                find, patch_rationale = _patch_fields(conn, sug_id)
                try:
                    patch_text(old, find or "", body or "", allow_empty=states_deletion(patch_rationale))
                except PatchError as exc:
                    print(f"  [{kind:<6}] {target_path}  (#{sug_id}) — deferred: {exc}")
                    continue
            noop = (kind == "index" and new_content == old)
            tag = "no-op" if noop else ("would apply" if dry else "applied")
            print(f"  [{kind:<6}] {target_path}  (#{sug_id}) — {tag}")
            if not dry:
                _apply_suggestion(conn, memory_root, sug_dir, sug_id, kind, target_path, new_content, sug_file, constitution_path=args.config.data['character']['constitution_path'])
        if not dry and _prune_orphaned_index_lines(memory_root, constitution_path=args.config.data['character']['constitution_path']):
            print("  pruned orphaned MEMORY.md links")
        return 0

    backup = _backup_memory(memory_root)
    print(f"{len(rows)} pending. Reviewing into {memory_root}; backup: {backup}\n")

    def drop_preview(name: str | None) -> None:
        """Remove the .suggestions/ preview file once a row is resolved (None for pre-migration rows)."""
        if not name:
            return
        f = sug_dir / name
        if f.exists():
            f.unlink()

    for sug_id, kind, target_path, body, rationale, sources, sug_file in rows:
        tgt = memory_root / target_path
        old = tgt.read_text(encoding="utf-8") if tgt.exists() else ""
        # `index` merges into MEMORY.md (append-only); all other kinds replace the file wholesale.
        new_content = _resolved_body(kind, old, body)
        print(_color("=" * 78, "grey"))
        print(f"#{sug_id}  [{_color(kind, 'yellow')}]  {target_path}")
        print(f"rationale: {rationale}")
        print(f"sources:   {sources}")
        print(_color("-" * 78, "grey"))

        shown = new_content
        if kind == "patch":
            find = _find_text(conn, sug_id)
            print(f"find: {find}")
            try:
                shown = patch_text(old, find or "", body or "", allow_empty=states_deletion(rationale))
            except PatchError as exc:
                print(_color(f"(patch cannot apply as proposed: {exc}; accepting defers it)", "red"))
                shown = None
        if kind == "remove":
            print(_color("(would delete this file)", "red"))
        elif shown is not None:
            # Diff against what we'll actually write (merged result for index, patched
            # file for patch), so the preview never lies about the outcome.
            diff = difflib.unified_diff(
                old.splitlines(keepends=True),
                shown.splitlines(keepends=True),
                fromfile=f"a/{target_path}",
                tofile=f"b/{target_path}",
                n=3,
            )
            for line in diff:
                if line.startswith("+++") or line.startswith("---"):
                    print(_color(line.rstrip(), "bold"))
                elif line.startswith("+"):
                    print(_color(line.rstrip(), "green"))
                elif line.startswith("-"):
                    print(_color(line.rstrip(), "red"))
                elif line.startswith("@@"):
                    print(_color(line.rstrip(), "cyan"))
                else:
                    print(line.rstrip())

        print(_color("-" * 78, "grey"))
        choice = input("[a]ccept / [r]eject / [s]kip / [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice == "a":
            if _apply_suggestion(conn, memory_root, sug_dir, sug_id, kind, target_path, new_content, sug_file, constitution_path=args.config.data['character']['constitution_path']):
                print(_color("  accepted.", "green"))
        elif choice == "r":
            conn.execute("UPDATE suggestions SET status='rejected', reviewed_at=CURRENT_TIMESTAMP WHERE id=?", (sug_id,))
            conn.commit()
            drop_preview(sug_file)
            print(_color("  rejected.", "red"))
        else:
            print(_color("  skipped.", "grey"))
    if _prune_orphaned_index_lines(memory_root, constitution_path=args.config.data['character']['constitution_path']):
        print(_color("  pruned orphaned MEMORY.md links", "grey"))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = open_db_readonly(args.db)
    def scalar(sql: str, *p: Any) -> int:
        return conn.execute(sql, p).fetchone()[0]
    print(f"DB:                 {args.db}")
    print(f"Projects:           {scalar('SELECT COUNT(DISTINCT project_slug) FROM sessions')}")
    print(f"Sessions ingested:  {scalar('SELECT COUNT(*) FROM sessions')}")
    print(f"Messages:           {scalar('SELECT COUNT(*) FROM messages'):,}")
    print(f"Distilled:          {scalar('SELECT COUNT(*) FROM distilled')}")
    pending = scalar("SELECT COUNT(*) FROM suggestions WHERE status='pending'")
    accepted = scalar("SELECT COUNT(*) FROM suggestions WHERE status='accepted'")
    print(f"Suggestions pending: {pending}")
    print(f"Suggestions accepted:{accepted}")
    last = conn.execute("SELECT MAX(created_at) FROM consolidation_runs WHERE run_id NOT LIKE 'audit-%'").fetchone()
    print(f"Last consolidate:   {last[0] if last and last[0] else 'never'}")
    audit = conn.execute("SELECT MAX(created_at) FROM consolidation_runs WHERE run_id LIKE 'audit-%'").fetchone()
    if audit and audit[0]:
        print(f"Last audit:         {audit[0]}")
    distill_route = args.config.stage("distill")
    consolidate_route = args.config.stage("consolidate")
    print(f"Distill route:      {distill_route['provider']} / {distill_route.get('model') or '(provider default)'}")
    print(f"Consolidate route:  {consolidate_route['provider']} / {consolidate_route.get('model') or '(provider default)'}")
    if getattr(args, "usage", False):
        _print_usage_report(conn, args.config.memory_root)
    return 0


def _memory_index_lines(memory_root: Path) -> dict[str, str]:
    try:
        lines = (memory_root / "MEMORY.md").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {}
    index: dict[str, str] = {}
    for line in lines:
        target = _index_target(line)
        if target and target not in index:
            index[target] = line.strip()
    return index


def _print_usage_report(conn: sqlite3.Connection, memory_root: Path) -> None:
    """Approved memory files automatic recall has not selected lately. Read-only."""
    import recall_usage

    try:
        report = recall_usage.usage_report(conn)
    except sqlite3.Error as exc:
        print(f"Recall usage:       not available ({type(exc).__name__}: recall tables unreadable)")
        return
    if report.state == "unavailable":
        print("Recall usage:       not available (database not migrated yet; the next `dream ingest` adds it)")
        return
    if report.state == "not-observed":
        print("Recall usage:       not observed yet (no recall hook markers harvested by `dream ingest`)")
        return
    days = recall_usage.UNUSED_DAYS
    print(f"Recall usage:       observed since {report.since} (harvested from recall hook markers)")
    print(f"  {len(report.unused)} of {report.approved_count} approved memory files not recalled in the last "
          f"{days} days (report only; nothing is deleted)")
    if report.window_shorter_than_days:
        print(f"  The observation window is shorter than {days} days, so recent files may not have had a chance.")
    index = _memory_index_lines(memory_root)
    usage = recall_usage.usage_map(conn)
    for entry in report.unused:
        print(f"  {entry.source_path} ({recall_usage.annotation(usage, entry.document_id)})")
        print(f"      {index.get(entry.source_path, '(not in MEMORY.md)')}")


def cmd_preflight(args: argparse.Namespace) -> int:
    failed = False
    for name, ok, detail in preflight(args.config):
        print(f"{'ok' if ok else 'FAIL'}  {name}: {detail}")
        failed = failed or not ok
    for name, ok, detail in recall_preflight(args.config, args.db):
        print(f"{'ok' if ok else 'FAIL'}  {name}: {detail}")
        failed = failed or not ok
    consolidate = args.config.consolidate_settings()
    mode = consolidate.mode if consolidate.mode == "full" else (
        f"{consolidate.mode} ({consolidate.subset_files} topic files; the weekly dream audit covers the whole store)")
    print(f"ok  consolidate.mode: {mode}")
    return 1 if failed else 0


def cmd_embed(args: argparse.Namespace) -> int:
    embedder, problem = configured_embedder(args.config)
    if problem:
        print(f"dream embed: embedder misconfigured: {problem}", file=sys.stderr)
        return 2
    if embedder is None:
        print('dream embed: the embedder is disabled; set [recall.embedder] enabled = true and type = "openai-compatible"',
              file=sys.stderr)
        return 2
    conn = open_db(args.db)
    try:
        # Synchronize first, so retention exclusions and edits are current
        # before any text leaves the process.
        if not _sync_recall_documents(conn, args.config.memory_root,
                                      constitution_path=args.config.data["character"]["constitution_path"]):
            return 1
        report = embed_recall_documents(
            conn, embedder, include_raw_transcripts=args.config.recall_settings().allow_raw_transcript_prompt,
            refresh=args.refresh,
        )
    finally:
        conn.close()
    print(f"Embedded {report.embedded} of {report.candidates} documents under {report.fingerprint}.")
    if report.error:
        print(f"dream embed: stopped: {report.error}", file=sys.stderr)
        return 1
    return 0


def _open_ephemeral_eval_db() -> sqlite3.Connection:
    """In-memory database for fixture runs: fixture rows never reach the live index."""
    conn = sqlite3.connect(":memory:")
    conn.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    _migrate(conn)
    return conn


def cmd_recall_eval(args: argparse.Namespace) -> int:
    from recall_eval import evaluate_fixture_file
    embedder, problem = configured_embedder(args.config)
    if problem:
        print(f"recall-eval: embedder misconfigured, reporting lexical only: {problem}", file=sys.stderr)
    conn = _open_ephemeral_eval_db()
    report = evaluate_fixture_file(args.fixtures, conn, args.config.recall_settings(), embedder=embedder)
    conn.close()
    result = report.to_dict()
    if getattr(args, "by_language", False):
        result["breakdown"] = report.breakdown_dict()
        # The headline metrics score the delivered context; these score selection before rendering.
        result["retrieval"] = report.retrieval
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_eval_build(args: argparse.Namespace) -> int:
    from eval_build import build_eval_fixture, collect_eval_documents, default_output_path
    output = Path(args.output).expanduser() if args.output else default_output_path()
    memory_root = args.config.memory_root
    conn = None
    try:
        # Read-only, and tolerant of a nightly writer's active WAL.
        conn = _open_curation_readonly(Path(args.db))
        _corpus, candidates = collect_eval_documents(conn, memory_root, config=args.config)
        if not candidates:
            print("eval-build: the recall index has no approved memories or distilled summaries to ask about; "
                  "run the nightly pipeline (dream distill) first.", file=sys.stderr)
            return 1
        if args.limit is None and not args.yes:
            print(f"Would ask the configured distill provider for questions about {len(candidates)} documents "
                  f"(one call each). Pass --limit N to cap, or --yes to proceed.")
            return 2
        fixture = build_eval_fixture(
            conn, memory_root, config=args.config, questions_per_file=args.questions_per_file,
            languages=args.languages, limit=args.limit, output=output,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"eval-build: {exc}", file=sys.stderr)
        return 2
    finally:
        if conn is not None:
            conn.close()
    generator = fixture["generator"]
    print(f"Wrote {len(fixture['queries'])} queries for {generator['sampled_documents']} documents "
          f"({len(fixture['documents'])} in the corpus) to {output}")
    print("This fixture holds private memory text: keep it out of the repository.")
    if generator["failed_documents"]:
        print(f"{generator['failed_documents']} documents failed at the provider and have no questions.", file=sys.stderr)
        return 1
    return 0


def cmd_recall_calibrate(args: argparse.Namespace) -> int:
    from recall_eval import EMBEDDER_MODE, calibrate_fixture_file
    embedder = None
    if args.mode == EMBEDDER_MODE:
        embedder, problem = configured_embedder(args.config)
        if embedder is None:
            print(f"recall-calibrate: {EMBEDDER_MODE} needs the configured embedder: {problem or 'the embedder is disabled'}",
                  file=sys.stderr)
            return 2
    apply = bool(getattr(args, "apply", False))
    # Calibrate on an in-memory copy of the fixture: a stale or now-excluded
    # fixture row must never be re-inserted into the live recall index.
    conn = _open_ephemeral_eval_db()
    try:
        record = calibrate_fixture_file(args.fixtures, conn, args.mode, embedder=embedder, require_usable=apply)
    except ValueError as exc:
        if not apply:
            raise
        print(f"recall-calibrate: {exc}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    result = {
        "mode": record.mode,
        "calibration_version": record.calibration_version,
        "threshold": record.threshold,
        "fixture_sha256": record.fixture_sha256,
        "created_at": record.created_at,
    }
    if apply:
        # The only writer of recall_calibrations: one row for the requested
        # mode, through the normal writer, after the evaluation succeeded.
        live = open_db(args.db)
        try:
            with live:
                live.execute(
                    "INSERT INTO recall_calibrations(mode, calibration_version, threshold, fixture_sha256, created_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(mode) DO UPDATE SET "
                    "calibration_version=excluded.calibration_version, threshold=excluded.threshold, "
                    "fixture_sha256=excluded.fixture_sha256, created_at=excluded.created_at",
                    (record.mode, record.calibration_version, record.threshold, record.fixture_sha256, record.created_at),
                )
        finally:
            live.close()
        result["applied_to"] = str(args.db)
        print(f"recall-calibrate: wrote recall_calibrations row mode={record.mode} threshold={record.threshold} "
              f"calibration_version={record.calibration_version} to {args.db}", file=sys.stderr)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_retention(args: argparse.Namespace) -> int:
    from dataclasses import replace
    from retention_cleanup import apply_retention
    from character.safety import guard_path
    config = replace(args.config, data={**args.config.data, "storage": {
        **args.config.data["storage"], "db_path": str(args.db), "memory_root": str(args.memory)}})
    guard_path(args.db, config)
    conn = open_db(args.db) if args.apply else open_db_readonly(args.db)
    try:
        result = apply_retention(conn, Path(args.memory), config=config, apply=args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["character"]["rerun_required"]:
            print("Character cleanup changed after backup. Rerun dream retention --apply.", file=sys.stderr)
            return 1
        return 0
    finally:
        conn.close()


def cmd_hooks(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if args.hooks_cmd == "install":
        report = install_hooks(path)
        print(f"Installed {report.installed} Dream automatic recall hooks in {path}.")
        print("Review and trust the exact commands through Codex /hooks before enabling automatic injection.")
        return 0
    report = uninstall_hooks(path)
    print(f"Removed {report.removed} Dream automatic recall hooks from {path}.")
    return 0


def _context_failure(args, exc: Exception, *, explain: bool) -> int:
    diagnostic = {"error": type(exc).__name__, "message": str(exc)[:400]}
    try:
        settings = args.config.recall_settings()
        append_diagnostic(settings, diagnostic)
    except Exception:
        pass
    if explain:
        print(json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":")))
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    explain = bool(getattr(args, "explain", False))
    conn = None
    attempt_token = None
    payload = {}
    try:
        payload = json.load(sys.stdin)
        event_name = "UserPromptSubmit" if args.context_cmd == "prompt" else "SessionStart"
        query = parse_hook_payload(event_name, payload)
        settings = args.config.recall_settings()
        conn = open_db_readonly(args.db)
        excluded = successful_session_start_ids(query.session_id) if args.context_cmd == "prompt" else frozenset()
        query = query.__class__(
            query.query_text, query.session_id, query.hook_event, query.cwd,
            query.repository_roots,
            settings.session_start_budget_codepoints if args.context_cmd == "session-start" else settings.prompt_budget_codepoints,
            excluded,
            settings.allow_raw_transcript_prompt if args.context_cmd == "prompt" else False,
        )
        if not settings.enabled:
            if not explain:
                print(hook_success_json(event_name, ""))
            return 0
        use_hook_state = query.hook_event != "prompt" or settings.first_prompt_only
        if use_hook_state and not (attempt_token := claim_hook_event(query.session_id, query.hook_event)):
            if explain:
                print(json.dumps({"candidate_count": 0, "selected_count": 0, "fallback_reason": "duplicate-or-running"}, separators=(",", ":")))
            return 0
        # Building the embedder contacts nothing; an unusable one means lexical recall.
        embedder, embedder_problem = configured_embedder(args.config)
        if embedder_problem:
            from recall_context import with_adapters
            settings = with_adapters(settings, embedder=False)
        result = run_context(conn, query, settings, explain=explain, embedder=embedder, allow_cache_writes=False,
                             memory_root=args.config.memory_root,
                             constitution_path=args.config.data["character"]["constitution_path"])
        fallback_reason = result.diagnostics.fallback_reason or ("embedder-misconfigured" if embedder_problem else None)
        if embedder_problem or str(fallback_reason).startswith(("embedder-unavailable", "embedder-error")):
            try:
                # One line per reason every ten minutes while the embedder stays down.
                if not recent_diagnostic(settings, fallback_reason):
                    append_diagnostic(settings, {"event": query.hook_event, "fallback_reason": fallback_reason,
                                                 "embedder_problem": (embedder_problem or "")[:400] or None})
            except Exception:
                pass
        character_text = ""
        character_diagnostic = None
        reminder_text = ""
        reminder_diagnostic = None
        if event_name == "SessionStart":
            try:
                from character.short import review_reminder
                reminder_text = review_reminder(conn, args.config)
            except Exception as reminder_error:
                reminder_diagnostic = "short-review-unavailable:" + type(reminder_error).__name__
        character_settings = args.config.data["character"]
        if event_name == "SessionStart" and character_settings["hook_injection"]:
            try:
                from character.render import render_block
                character_text = render_block(conn, args.config)["text"]
            except Exception as character_error:
                character_diagnostic = "character-unavailable:" + type(character_error).__name__
        from recall_context import compose_hook_context
        combined_context, budget_diagnostic = compose_hook_context(
            result.rendered_context, character_text, query.requested_codepoint_budget,
            settings.session_start_additional_context_limit if event_name == "SessionStart"
            else settings.prompt_additional_context_limit,
            reminder_text=reminder_text,
        )
        character_diagnostic = character_diagnostic or budget_diagnostic
        if attempt_token is not None:
            finish_hook_event(query.session_id, query.hook_event, succeeded=True, selected_ids=[candidate.document.id for candidate in result.selected], attempt_token=attempt_token)
        if explain:
            print(json.dumps({
                "candidate_ids": [candidate.document.id for candidate in result.selected],
                "selected_codepoints": len(combined_context),
                "character_diagnostic": character_diagnostic,
                "short_review_diagnostic": reminder_diagnostic,
                "elapsed_ms": result.diagnostics.elapsed_ms,
                "calibration_version": result.calibration_version,
                "fallback_reason": fallback_reason,
            }, ensure_ascii=False, separators=(",", ":")))
        else:
            print(hook_success_json(event_name, combined_context))
        return 0
    except Exception as exc:
        if attempt_token is not None:
            try:
                finish_hook_event(str(payload.get("session_id", "unknown")), getattr(locals().get("query", None), "hook_event", "prompt"), succeeded=False, error_code=type(exc).__name__, attempt_token=attempt_token)
            except Exception:
                pass
        return _context_failure(args, exc, explain=explain)


def cmd_mcp(args: argparse.Namespace) -> int:
    """``mcp serve`` speaks JSON-RPC on stdio; ``mcp install|uninstall`` edit one client's config.

    ``serve`` keeps stdout for protocol messages only and stays up when the
    configuration fails to load: every tool call then reports the error.
    """
    if args.mcp_cmd == "serve":
        from mcp_server import serve
        try:
            return serve(sys.stdin, sys.stdout, args.config, db_path=args.db,
                         config_error=getattr(args, "config_error", None))
        except KeyboardInterrupt:
            return 130
    import mcp_install
    if args.client == "claude" and args.path:
        print("dream mcp: --path applies to codex and opencode only; Claude Code manages its own file",
              file=sys.stderr)
        return 2
    path = Path(args.path).expanduser() if args.path else None
    launch = mcp_install.server_launch()
    try:
        if args.mcp_cmd == "install":
            report = mcp_install.install(args.client, path, launch)
        else:
            report = mcp_install.uninstall(args.client, path)
    except mcp_install.McpInstallError as exc:
        print(f"dream mcp: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"dream mcp: {exc.strerror or type(exc).__name__}", file=sys.stderr)
        return 1
    where = f" in {report.path}" if report.path else ""
    if args.mcp_cmd == "install":
        if not report.changed:
            print(f"The Dream MCP server is already installed for {args.client}{where}.")
        elif report.detail == "updated":
            print(f"Updated the Dream MCP server entry for {args.client}{where}.")
        else:
            print(f"Installed the Dream MCP server for {args.client}{where}.")
    else:
        print(f"Removed the Dream MCP server for {args.client}{where}." if report.changed
              else f"Nothing to remove for {args.client}{where}: {report.detail}.")
    for note in report.notes:
        print(note)
    if report.backup_path:
        print(f"Backup of the previous file: {report.backup_path}")
    if args.mcp_cmd == "install":
        print(mcp_install.next_step(args.client, launch))
    return 0


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _recall_kinds(value: str) -> tuple[str, ...]:
    kinds = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [kind for kind in kinds if kind not in RECALL_KINDS]
    if not kinds or unknown:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of: {', '.join(RECALL_KINDS)}")
    return kinds


def _language_codes(value: str) -> tuple[str, ...]:
    from eval_build import validate_languages
    try:
        return validate_languages(part for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dream", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="TOML configuration path (env: DREAM_CONFIG)")
    p.add_argument("--db", help="SQLite DB path; overrides configuration")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Scan configured transcript sources into SQLite")
    pi.add_argument("--projects", help="Legacy one-off Claude projects root")
    pi.add_argument("--source", action="append", help="Configured source name; repeatable")
    pi.add_argument("--force", action="store_true", help="Re-ingest even if content revision is unchanged")
    pi.add_argument("--verbose", "-v", action="store_true")
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("estimate", help="Estimate subscription-limit draw of distilling pending sessions")
    pe.add_argument("--min-chars", type=int, default=500)
    pe.add_argument("--project", help="Limit to one project slug")
    pe.add_argument("--verbose", "-v", action="store_true")
    pe.add_argument("--refresh", action="store_true", help="Include sessions produced by older prompt/provider configuration")
    pe.set_defaults(func=cmd_estimate)

    pd = sub.add_parser("distill", help="Run the configured distillation stage")
    pd.add_argument("--min-chars", type=int, default=500)
    pd.add_argument("--project", help="Limit to one project slug")
    pd.add_argument("--limit", type=int, help="Process at most N sessions")
    pd.add_argument("--max-sessions", type=int, default=5, help="Confirm with --yes if more than this")
    pd.add_argument("--yes", "-y", action="store_true")
    pd.add_argument("--model", default=None, help="Temporary model override")
    pd.add_argument("--refresh", action="store_true", help="Re-distill sessions produced by older prompt/provider configuration")
    pd.set_defaults(func=cmd_distill)

    pc = sub.add_parser("consolidate", help="Run the configured consolidation stage")
    pc.add_argument("--memory")
    pc.add_argument("--since", help="Optional legacy timestamp lower bound")
    pc.add_argument("--max-sessions", type=int, default=200)
    pc.add_argument("--model", default=None, help="Temporary model override")
    pc.add_argument("--mode", choices=["full", "subset"], default=None,
                    help="Override consolidate.mode for this run")
    pc.set_defaults(func=cmd_consolidate)

    pa = sub.add_parser("audit", help="Weekly staleness audit over the whole store (remove/patch suggestions)")
    pa.add_argument("--memory")
    pa.add_argument("--model", default=None, help="Temporary model override")
    pa.add_argument("--force", action="store_true",
                    help="Audit even when consolidate.mode is full (the nightly run already audits)")
    pa.set_defaults(func=cmd_audit)

    ps = sub.add_parser("search", help="FTS5 search over transcripts")
    ps.add_argument("query")
    ps.add_argument("--limit", type=int, default=20)
    ps.add_argument("--role", choices=["user", "assistant"])
    ps.add_argument("--project")
    ps.set_defaults(func=cmd_search)

    prc = sub.add_parser("recall", help="Read-only approved-memory query for agents (untrusted reference data)")
    prc.add_argument("query", help="3-8 content words, passed as one argument")
    prc.add_argument("--project", help="Prefer this project slug; other projects stay eligible")
    prc.add_argument("--kinds", type=_recall_kinds,
                     help="Comma-separated source kinds (default approved_memory,distilled_summary; "
                          "raw_transcript must be named and needs recall.allow_raw_transcript_prompt)")
    prc.add_argument("--limit", type=int, default=5, help="Results to return, clamped to 1-10 (default 5)")
    prc.add_argument("--json", action="store_true",
                     help="Print one JSON object with mode, calibration version, results and the rendered context")
    prc.set_defaults(func=cmd_recall)

    pr = sub.add_parser("review", help="Walk pending suggestions interactively")
    pr.add_argument("--memory")
    pr.add_argument("--yes", action="store_true",
                    help="Accept every pending suggestion non-interactively (backs up memory first). Nightly path.")
    pr.add_argument("--dry-run", action="store_true",
                    help="With --yes, print the apply plan without writing anything.")
    pr.set_defaults(func=cmd_review)

    psg = sub.add_parser("suggestions", help="Machine-readable suggestion review")
    psg.add_argument("--memory")
    ssub = psg.add_subparsers(dest="suggestions_cmd", required=True)
    slist = ssub.add_parser("list", help="List pending suggestions as JSON")
    slist.set_defaults(func=cmd_suggestions)
    for action in ("accept", "reject"):
        parser = ssub.add_parser(action)
        parser.add_argument("id", type=int)
        parser.set_defaults(func=cmd_suggestions)
    smerge = ssub.add_parser("merge")
    smerge.add_argument("id", type=int)
    smerge.add_argument("--body-file", required=True)
    smerge.set_defaults(func=cmd_suggestions)
    sall = ssub.add_parser("apply-all")
    sall.set_defaults(func=cmd_suggestions)
    sconfigured = ssub.add_parser("apply-configured")
    sconfigured.set_defaults(func=cmd_suggestions)
    scurate = ssub.add_parser("curate-configured")
    scurate.add_argument("--dry-run", action="store_true")
    scurate.set_defaults(func=cmd_suggestions)

    pst = sub.add_parser("status", help="Show DB stats")
    pst.add_argument("--usage", action="store_true",
                     help="List approved memory files not recalled in the last 90 days (read-only)")
    pst.set_defaults(func=cmd_status)

    pret = sub.add_parser("retention", help="Preview or apply local memory exclusions")
    pret.add_argument("--memory")
    pret.add_argument("--apply", action="store_true", help="Back up and remove excluded generated memories")
    pret.set_defaults(func=cmd_retention)

    ppre = sub.add_parser("preflight", help="Check configured provider executables and auth")
    ppre.set_defaults(func=cmd_preflight)

    pemb = sub.add_parser("embed", help="Store recall embeddings with the configured local embedder")
    pemb.add_argument("--refresh", action="store_true",
                      help="Re-embed every current document, e.g. after a model change behind the same name")
    pemb.set_defaults(func=cmd_embed)

    pctx = sub.add_parser("context", help="Run fail-open automatic recall context")
    csub = pctx.add_subparsers(dest="context_cmd", required=True)
    for context_cmd in ("session-start", "prompt"):
        cp = csub.add_parser(context_cmd)
        cp.add_argument("--explain", action="store_true")
        cp.set_defaults(func=cmd_context)

    peval = sub.add_parser("recall-eval", help="Evaluate a recall fixture")
    peval.add_argument("--fixtures", required=True)
    peval.add_argument("--by-language", action="store_true",
                       help="Add Recall@1/@5, MRR, forbidden results and p95 latency per mode and language")
    peval.set_defaults(func=cmd_recall_eval)
    pbuild = sub.add_parser("eval-build", help="Build a private held-out recall fixture with the distill provider")
    pbuild.add_argument("--limit", type=_positive_int, help="Ask about at most N documents")
    pbuild.add_argument("--questions-per-file", type=_positive_int, default=3,
                        help="Questions per document and language (default 3)")
    pbuild.add_argument("--languages", type=_language_codes, default=("en", "pl"),
                        help="Comma-separated language codes (default en,pl)")
    pbuild.add_argument("--output", help="Fixture path (default ~/.config/dream/eval/private.json)")
    pbuild.add_argument("--yes", "-y", action="store_true", help="Proceed without --limit")
    pbuild.set_defaults(func=cmd_eval_build)
    pcal = sub.add_parser("recall-calibrate", help="Calibrate recall thresholds")
    pcal.add_argument("--fixtures", required=True)
    pcal.add_argument("--mode", required=True)
    pcal.add_argument("--output")
    pcal.add_argument("--apply", action="store_true",
                      help="Write the calibration for --mode to the live database (recall_calibrations)")
    pcal.set_defaults(func=cmd_recall_calibrate)

    phooks = sub.add_parser("hooks", help="Install or remove Codex recall hooks")
    hsub = phooks.add_subparsers(dest="hooks_cmd", required=True)
    for hooks_cmd in ("install", "uninstall"):
        hp = hsub.add_parser(hooks_cmd)
        hp.add_argument("--path", default=os.environ.get("DREAM_CODEX_HOOKS_PATH", str(Path.home() / ".codex" / "hooks.json")))
        hp.set_defaults(func=cmd_hooks)

    pmcp = sub.add_parser("mcp", help="Read-only recall tools for agents over MCP stdio, and client installers")
    msub = pmcp.add_subparsers(dest="mcp_cmd", required=True)
    mserve = msub.add_parser("serve", help="Serve recall_search, recall_get and memory_index as JSON-RPC on stdio")
    mserve.set_defaults(func=cmd_mcp)
    for mcp_cmd, verb in (("install", "Add"), ("uninstall", "Remove")):
        mp = msub.add_parser(mcp_cmd, help=f"{verb} the dream MCP server entry in one client's configuration")
        mp.add_argument("--client", required=True, choices=("codex", "opencode", "claude"))
        mp.add_argument("--path", help="Client config file for codex or opencode (default: the client's global config)")
        mp.set_defaults(func=cmd_mcp)

    from character.cli import register as register_character
    register_character(sub)

    args = p.parse_args(argv)
    try:
        args.config = load_config(Path(args.config).expanduser() if args.config else None)
    except Exception as exc:
        if args.cmd == "context":
            if getattr(args, "explain", False):
                print(json.dumps({"error": type(exc).__name__, "message": str(exc)[:400]}, separators=(",", ":")))
            return 0
        if args.cmd == "recall":
            return _recall_unavailable(f"configuration error: {type(exc).__name__}")
        if args.cmd != "mcp":
            raise
        # mcp serve keeps speaking JSON-RPC and reports the error per tool call; install needs no config.
        args.config, args.config_error = None, type(exc).__name__
        args.db = Path(args.db).expanduser() if args.db else None
        return args.func(args)
    args.db = Path(args.db).expanduser() if args.db else args.config.db_path
    if hasattr(args, "memory"):
        args.memory = str(Path(args.memory).expanduser()) if args.memory else str(args.config.memory_root)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
