#!/usr/bin/env python3
"""
dream — provider-agnostic local memory consolidation.

Subcommands:
  ingest        Scan configured transcript sources into SQLite
  estimate      Show how many sessions would be distilled + subscription-limit draw
  distill       Run the configured distillation stage and store JSON notes
  consolidate   Run the configured consolidation stage and propose updates
  search        FTS5 search over all ingested transcripts
  review        Walk pending suggestions, accept/reject interactively
  status        Show DB stats

Safe defaults:
  * `ingest` is idempotent (mtime-checked)
  * `distill` requires --yes once you exceed --max-sessions (default 5)
  * `consolidate` writes only to memory/.suggestions/, never to live memory files
  * only explicit review/apply commands touch user-curated memory
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
from typing import Any

# resolve() so the installed launcher (~/.local/bin/dream is a symlink into the repo)
# finds schema.sql / prompts next to the real file, not next to the symlink.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ingest import discover_jsonl, home_project_slug, ingest_file, ingest_configured_sources  # noqa: E402
from distill import distill_session, sessions_needing_distill  # noqa: E402
from consolidate import consolidate as run_consolidate  # noqa: E402
from search import search as fts_search, session_summary  # noqa: E402
from backend import active_backend, generate, preflight  # noqa: E402
from config import load_config  # noqa: E402
from curate import CURATION_SCHEMA, build_curation_prompt, parse_curation_output  # noqa: E402
from recall_documents import synchronize_recall_documents  # noqa: E402
from recall_context import append_diagnostic, parse_hook_payload, recall_preflight, run_context, hook_success_json  # noqa: E402
from recall_hook_state import claim_hook_event, finish_hook_event, successful_session_start_ids  # noqa: E402
from recall_hooks import install_hooks, uninstall_hooks  # noqa: E402
from storage import open_db_readonly  # noqa: E402


def _sync_recall_documents(conn: sqlite3.Connection, memory_root,
                           *, include_raw_transcripts: bool = False) -> None:
    """Keep the recall store current after a successful mutation.

    Synchronization is quiet on success and non-fatal: any error (missing
    recall objects, FTS problems, ...) is reported to stderr and swallowed so
    the caller's already-committed mutation is never undone and the command
    keeps its documented behavior. Raw transcripts are opt-in.
    """
    try:
        synchronize_recall_documents(
            conn,
            memory_root,
            include_raw_transcripts=include_raw_transcripts,
        )
    except Exception as e:
        print(f"recall sync skipped: {type(e).__name__}: {e}", file=sys.stderr)


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
    )
    if not needs_upgrade:
        return None
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = path.with_name(f"{path.name}.bak-provider-migration-{stamp}")
    if not backup_path.exists():
        dest = sqlite3.connect(backup_path)
        try:
            conn.backup(dest)
        finally:
            dest.close()
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


def _prune_orphaned_index_lines(memory_root: Path) -> bool:
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
    lines = index_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [
        line for line in lines
        if (t := _index_target(line)) is None or (memory_root / t).exists()
    ]
    if kept == lines:
        return False
    index_path.write_text("".join(kept), encoding="utf-8")
    return True


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = open_db(args.db)
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
    _sync_recall_documents(conn, args.config.memory_root)
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    conn = open_db_readonly(args.db)
    candidates = sessions_needing_distill(
        conn, min_chars=args.min_chars, project_slug=args.project, config=args.config,
        refresh_config=getattr(args, "refresh", False),
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
        refresh_config=getattr(args, "refresh", False),
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


def cmd_consolidate(args: argparse.Namespace) -> int:
    conn = open_db(args.db)
    memory_root = Path(args.memory)
    memory_root.mkdir(parents=True, exist_ok=True)
    print(f"Consolidating into suggestions for {memory_root}")
    suggestions = run_consolidate(
        conn, memory_root,
        since_iso=args.since,
        model=args.model,
        max_sessions=args.max_sessions,
        config=args.config,
    )
    if not suggestions:
        print("No suggestions produced.")
        return 0
    print(f"\nWrote {len(suggestions)} suggestions to {memory_root}/.suggestions/")
    for s in suggestions:
        kind_color = {"new": "green", "update": "yellow", "remove": "red", "index": "blue"}.get(s.kind, "grey")
        print(f"  [{_color(s.kind, kind_color):>14}] {s.target_path}  — {s.rationale[:80]}")
    print(f"\nRun `dream review` to walk through them.")
    return 0


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


def _apply_suggestion(conn, memory_root: Path, sug_dir: Path,
                      sug_id, kind, target_path, new_content, sug_file,
                      sync_recall: bool = True) -> bool:
    """Accept one suggestion: write/merge/delete the target, mark accepted, drop the preview.

    Refuses (marks 'rejected' instead of touching the filesystem) if target_path would
    resolve outside memory_root, or if a 'remove' suggestion targets MEMORY.md itself
    (the index file, never deletable via this path). target_path comes from LLM output
    and is otherwise unvalidated before hitting unlink()/write_text().
    """
    tgt = memory_root / target_path
    contained = tgt.resolve().is_relative_to(memory_root.resolve())
    protected = kind == "remove" and target_path == "MEMORY.md"
    state = conn.execute(
        "SELECT base_sha256, target_existed FROM suggestions WHERE id=?", (sug_id,)
    ).fetchone()
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
    if contained and not protected:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if kind == "remove" and tgt.exists():
            tgt.unlink()
        elif kind in ("new", "update", "index"):
            tgt.write_text(new_content, encoding="utf-8")
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
        _sync_recall_documents(conn, memory_root)
    return status == "accepted"


def _pending_suggestions(conn: sqlite3.Connection) -> list[sqlite3.Row | tuple]:
    return conn.execute(
        "SELECT id, kind, target_path, body, rationale, source_sessions, sug_file, "
        "base_sha256, target_existed FROM suggestions "
        "WHERE status='pending' ORDER BY id"
    ).fetchall()


def _safe_suggestion_target(memory_root: Path, target_path: str) -> Path | None:
    """Resolve a suggestion target only when it remains inside memory_root."""
    root = memory_root.resolve()
    target = (memory_root / target_path).resolve()
    return target if target.is_relative_to(root) else None


def _curation_target_snapshot(memory_root: Path, row: tuple) -> dict[str, Any]:
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


def _cmd_curate_configured(args: argparse.Namespace, conn, memory_root: Path, rows) -> int:
    """Curate one complete pending batch through the configured review provider."""
    if args.config.data["review"]["mode"] != "auto-apply":
        print("review.mode is suggest-only; curation disabled, leaving suggestions pending")
        return 0
    if not rows:
        print("No pending suggestions.")
        return 0

    try:
        snapshots = [_curation_target_snapshot(memory_root, row) for row in rows]
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
    if dry_run:
        print("dry-run: no files, previews, statuses, or backups changed")
        return 0

    by_id = {snapshot["id"]: snapshot for snapshot in snapshots}
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
        if kind == "remove" and target_path == "MEMORY.md":
            _reject_suggestion(conn, memory_root / ".suggestions", sug_id, sug_file)
            rejected.append(sug_id)
            unsafe.append(sug_id)
            print(f"unsafe #{sug_id} ({target_path}): protected target; rejected by host safety check")
            continue
        if snapshot["conflict"] and decision.decision == "accept":
            deferred.append(sug_id)
            conflicts.append(sug_id)
            print(f"conflict #{sug_id} ({target_path}): initial conflict; accept deferred")
            continue

        rechecked_target, old, current_sha, target_existed = _curation_recheck(memory_root, snapshot)
        if (
            rechecked_target is None
            or current_sha != snapshot["current_sha256"]
            or target_existed != snapshot["target_existed"]
        ):
            deferred.append(sug_id)
            conflicts.append(sug_id)
            print(f"conflict #{sug_id} ({target_path}): target changed during curation")
            continue

        if decision.decision == "merge" and snapshot["conflict"]:
            with conn:
                conn.execute(
                    "UPDATE suggestions SET base_sha256=?, target_existed=? WHERE id=?",
                    (current_sha, int(target_existed), sug_id),
                )
        proposed_body = snapshot["body"] if decision.decision == "accept" else decision.body
        new_content = _resolved_body(kind, old, proposed_body)
        mutates_file = (
            (kind == "remove" and target_existed)
            or (kind in ("new", "update", "index") and (not target_existed or new_content != old))
        )
        if mutates_file and backup is None:
            backup = _backup_memory(memory_root)
        try:
            applied_ok = _apply_suggestion(
                conn,
                memory_root,
                memory_root / ".suggestions",
                sug_id,
                kind,
                target_path,
                new_content,
                sug_file,
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
        elif sug_id not in conflicts:
            deferred.append(sug_id)
            conflicts.append(sug_id)

    if applied and _prune_orphaned_index_lines(memory_root):
        print("pruned orphaned MEMORY.md links")
    print(f"applied IDs: {applied}")
    print(f"rejected IDs: {rejected}")
    print(f"deferred IDs: {deferred}")
    print(f"conflict IDs: {conflicts}")
    print(f"backup: {backup if backup is not None else 'none'}")
    if unsafe:
        print(f"unsafe IDs: {unsafe}", file=sys.stderr)
    return 1 if unsafe else 0


def cmd_suggestions(args: argparse.Namespace) -> int:
    conn = open_db_readonly(args.db) if args.suggestions_cmd == "list" else open_db(args.db)
    memory_root = Path(args.memory)
    sug_dir = memory_root / ".suggestions"
    rows = _pending_suggestions(conn)

    if args.suggestions_cmd == "curate-configured":
        return _cmd_curate_configured(args, conn, memory_root, rows)

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
            if not _apply_suggestion(conn, memory_root, sug_dir, sug_id, kind, target, new_content, preview):
                conflicts += 1
        if _prune_orphaned_index_lines(memory_root):
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
    new_content = _resolved_body(kind, old, body)
    backup = _backup_memory(memory_root)
    print(f"Backup: {backup}")
    return 0 if _apply_suggestion(
        conn, memory_root, sug_dir, sug_id, kind, target, new_content, preview
    ) else 3


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
            noop = (kind == "index" and new_content == old)
            tag = "no-op" if noop else ("would apply" if dry else "applied")
            print(f"  [{kind:<6}] {target_path}  (#{sug_id}) — {tag}")
            if not dry:
                _apply_suggestion(conn, memory_root, sug_dir, sug_id, kind, target_path, new_content, sug_file)
        if not dry and _prune_orphaned_index_lines(memory_root):
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

        if kind == "remove":
            print(_color("(would delete this file)", "red"))
        else:
            # Diff against what we'll actually write (merged result for index), so the
            # preview never lies about the outcome.
            diff = difflib.unified_diff(
                old.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
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
            _apply_suggestion(conn, memory_root, sug_dir, sug_id, kind, target_path, new_content, sug_file)
            print(_color("  accepted.", "green"))
        elif choice == "r":
            conn.execute("UPDATE suggestions SET status='rejected', reviewed_at=CURRENT_TIMESTAMP WHERE id=?", (sug_id,))
            conn.commit()
            drop_preview(sug_file)
            print(_color("  rejected.", "red"))
        else:
            print(_color("  skipped.", "grey"))
    if _prune_orphaned_index_lines(memory_root):
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
    last = conn.execute("SELECT MAX(created_at) FROM consolidation_runs").fetchone()
    print(f"Last consolidate:   {last[0] if last and last[0] else 'never'}")
    distill_route = args.config.stage("distill")
    consolidate_route = args.config.stage("consolidate")
    print(f"Distill route:      {distill_route['provider']} / {distill_route.get('model') or '(provider default)'}")
    print(f"Consolidate route:  {consolidate_route['provider']} / {consolidate_route.get('model') or '(provider default)'}")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    failed = False
    for name, ok, detail in preflight(args.config):
        print(f"{'ok' if ok else 'FAIL'}  {name}: {detail}")
        failed = failed or not ok
    for name, ok, detail in recall_preflight(args.config, args.db):
        print(f"{'ok' if ok else 'FAIL'}  {name}: {detail}")
        failed = failed or not ok
    return 1 if failed else 0


def cmd_recall_eval(args: argparse.Namespace) -> int:
    from recall_eval import evaluate_fixture_file
    conn = sqlite3.connect(":memory:")
    conn.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    _migrate(conn)
    report = evaluate_fixture_file(args.fixtures, conn, args.config.recall_settings())
    conn.close()
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


def cmd_recall_calibrate(args: argparse.Namespace) -> int:
    from recall_eval import calibrate_fixture_file
    record = calibrate_fixture_file(args.fixtures, open_db(args.db), args.mode)
    result = {
        "mode": record.mode,
        "calibration_version": record.calibration_version,
        "threshold": record.threshold,
        "fixture_sha256": record.fixture_sha256,
        "created_at": record.created_at,
    }
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


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
                print(hook_success_json(""))
            return 0
        use_hook_state = query.hook_event != "prompt" or settings.first_prompt_only
        if use_hook_state and not (attempt_token := claim_hook_event(query.session_id, query.hook_event)):
            if explain:
                print(json.dumps({"candidate_count": 0, "selected_count": 0, "fallback_reason": "duplicate-or-running"}, separators=(",", ":")))
            return 0
        result = run_context(conn, query, settings, explain=explain, allow_cache_writes=False)
        if attempt_token is not None:
            finish_hook_event(query.session_id, query.hook_event, succeeded=True, selected_ids=[candidate.document.id for candidate in result.selected], attempt_token=attempt_token)
        if explain:
            print(json.dumps({
                "candidate_ids": [candidate.document.id for candidate in result.selected],
                "selected_codepoints": len(result.rendered_context),
                "elapsed_ms": result.diagnostics.elapsed_ms,
                "calibration_version": result.calibration_version,
                "fallback_reason": result.diagnostics.fallback_reason,
            }, ensure_ascii=False, separators=(",", ":")))
        else:
            print(hook_success_json(result.rendered_context))
        return 0
    except Exception as exc:
        if attempt_token is not None:
            try:
                finish_hook_event(str(payload.get("session_id", "unknown")), getattr(locals().get("query", None), "hook_event", "prompt"), succeeded=False, error_code=type(exc).__name__, attempt_token=attempt_token)
            except Exception:
                pass
        return _context_failure(args, exc, explain=explain)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dream", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="TOML configuration path (env: DREAM_CONFIG)")
    p.add_argument("--db", help="SQLite DB path; overrides configuration")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Scan JSONL transcripts into DB")
    pi.add_argument("--projects", help="Legacy one-off Claude projects root")
    pi.add_argument("--source", action="append", help="Configured source name; repeatable")
    pi.add_argument("--force", action="store_true", help="Re-ingest even if mtime unchanged")
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
    pc.set_defaults(func=cmd_consolidate)

    ps = sub.add_parser("search", help="FTS5 search over transcripts")
    ps.add_argument("query")
    ps.add_argument("--limit", type=int, default=20)
    ps.add_argument("--role", choices=["user", "assistant"])
    ps.add_argument("--project")
    ps.set_defaults(func=cmd_search)

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
    pst.set_defaults(func=cmd_status)

    ppre = sub.add_parser("preflight", help="Check configured provider executables and auth")
    ppre.set_defaults(func=cmd_preflight)

    pctx = sub.add_parser("context", help="Run fail-open automatic recall context")
    csub = pctx.add_subparsers(dest="context_cmd", required=True)
    for context_cmd in ("session-start", "prompt"):
        cp = csub.add_parser(context_cmd)
        cp.add_argument("--explain", action="store_true")
        cp.set_defaults(func=cmd_context)

    peval = sub.add_parser("recall-eval", help="Evaluate a recall fixture")
    peval.add_argument("--fixtures", required=True)
    peval.set_defaults(func=cmd_recall_eval)
    pcal = sub.add_parser("recall-calibrate", help="Calibrate recall thresholds")
    pcal.add_argument("--fixtures", required=True)
    pcal.add_argument("--mode", required=True)
    pcal.add_argument("--output")
    pcal.set_defaults(func=cmd_recall_calibrate)

    phooks = sub.add_parser("hooks", help="Install or remove Codex recall hooks")
    hsub = phooks.add_subparsers(dest="hooks_cmd", required=True)
    for hooks_cmd in ("install", "uninstall"):
        hp = hsub.add_parser(hooks_cmd)
        hp.add_argument("--path", default=os.environ.get("DREAM_CODEX_HOOKS_PATH", str(Path.home() / ".codex" / "hooks.json")))
        hp.set_defaults(func=cmd_hooks)

    args = p.parse_args(argv)
    try:
        args.config = load_config(Path(args.config).expanduser() if args.config else None)
    except Exception as exc:
        if args.cmd == "context":
            if getattr(args, "explain", False):
                print(json.dumps({"error": type(exc).__name__, "message": str(exc)[:400]}, separators=(",", ":")))
            return 0
        raise
    args.db = Path(args.db).expanduser() if args.db else args.config.db_path
    if hasattr(args, "memory"):
        args.memory = str(Path(args.memory).expanduser()) if args.memory else str(args.config.memory_root)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
