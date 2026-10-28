"""Apply user-requested memory exclusions without touching source transcripts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import uuid

from recall_documents import synchronize_recall_documents
from retention_policy import empty_notes, excludes_suggestion, filter_notes, load_retention_policy


def _patch_preview_text(root: Path, sug_file: str | None) -> str:
    """A pending patch's preview: its unified diff's context lines can copy text outside the span."""
    if not sug_file:
        return ""
    preview = root / ".suggestions" / sug_file
    if not preview.resolve().is_relative_to(root / ".suggestions"):
        return ""
    try:
        return preview.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _rows(conn, sql: str, args) -> list[dict]:
    cursor = conn.execute(sql, args)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def apply_retention(conn, memory_root: Path, *, config, apply=False) -> dict:
    from character.retention import collect, preview, apply_snapshot
    character_snapshot = collect(conn, config)
    from character.safety import is_memory_control_source
    def guard_control(path):
        if is_memory_control_source(path, memory_root=memory_root,
                                    constitution_path=config.data['character']['constitution_path']):
            raise ValueError("retention target aliases a protected input")
    root = memory_root.resolve()
    guard_control(root / 'MEMORY.md')
    policy = load_retention_policy(root)
    files = []
    for name in policy.memory_paths:
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("excluded memory resolves outside the memory root")
        if path.is_file():
            guard_control(path)
            files.append(path)

    forgotten = [sid for sid in policy.session_ids
                 if conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (sid,)).fetchone()]

    changed = []
    for sid, project, notes_json, summary in conn.execute(
        "SELECT d.session_id, s.project_slug, d.notes_json, d.summary "
        "FROM distilled d JOIN sessions s ON s.session_id=d.session_id"
    ):
        if sid in forgotten:
            continue
        notes = json.loads(notes_json)
        filtered = empty_notes() if policy.excludes(project_slug=project) else filter_notes(notes, policy)
        filtered_summary = "" if policy.excludes(project_slug=project, text=summary or "") else summary
        if filtered != notes or filtered_summary != summary:
            changed.append((sid, notes_json, summary, json.dumps(filtered, ensure_ascii=False), filtered_summary))

    # A database not yet migrated (preview runs read-only) has no patch spans.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(suggestions)")}
    find_text = "find_text" if "find_text" in columns else "NULL"
    pending = [row for row in conn.execute(
        f"SELECT id, target_path, body, rationale, sug_file, source_sessions, {find_text}, kind "
        "FROM suggestions WHERE status='pending'"
    ).fetchall() if excludes_suggestion(
        conn, policy, root, target_path=row[1], source_sessions=row[5],
        texts=(row[2], row[3], row[6], _patch_preview_text(root, row[4]) if row[7] == "patch" else None))]
    result = {"memory_files": [str(p.relative_to(root)) for p in files],
              "distilled_records": len(changed), "sessions": len(forgotten), "pending_suggestions": len(pending), "applied": apply, "character": preview(character_snapshot)}
    if not apply:
        return result

    # Backup only affected records, outside every automatic-memory source directory.
    backup = root.parent / "retention-backups" / (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    backup.mkdir(parents=True, mode=0o700)
    snapshots = {str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in files}
    index = root / "MEMORY.md"
    if index.exists():
        if not index.resolve().is_relative_to(root):
            raise ValueError("memory index resolves outside the memory root")
        snapshots["MEMORY.md"] = index.read_text(encoding="utf-8")
    backup_file = backup / "affected-records.json"
    with backup_file.open("x", encoding="utf-8") as handle:
        backup_file.chmod(0o600)
        sessions = [{
            "session": _rows(conn, "SELECT * FROM sessions WHERE session_id=?", (sid,))[0],
            "messages": _rows(conn, "SELECT * FROM messages WHERE session_id=? ORDER BY seq", (sid,)),
            "distilled": _rows(conn, "SELECT * FROM distilled WHERE session_id=?", (sid,)),
        } for sid in forgotten]
        json.dump({"files": snapshots, "distilled": [row[:3] for row in changed], "sessions": sessions,
                   "pending_suggestions": pending, "character": character_snapshot}, handle, ensure_ascii=False, indent=2)

    result["character"] = apply_snapshot(conn, config, character_snapshot)

    # Repeatable cleanup. The active policy already prevents recall if interrupted.
    deferred_files = []
    deferred_sessions = []
    for path in files:
        name = str(path.relative_to(root))
        if not path.exists() or path.read_text(encoding="utf-8") != snapshots[name]:
            deferred_files.append(name)
            continue
        guard_control(path)
        path.unlink()
    if "MEMORY.md" in snapshots:
        lines = snapshots["MEMORY.md"].splitlines(keepends=True)
        excluded_paths = set(policy.memory_paths) - set(deferred_files)
        kept = "".join(line for line in lines if not any(f"({name})" in line for name in excluded_paths))
        if kept != snapshots["MEMORY.md"]:
            if not index.exists() or index.read_text(encoding="utf-8") != snapshots["MEMORY.md"]:
                deferred_files.append("MEMORY.md")
            else:
                guard_control(index)
                index.write_text(kept, encoding="utf-8")
    with conn:
        for sid, old_notes, old_summary, notes, summary in changed:
            updated = conn.execute(
                "UPDATE distilled SET notes_json=?, summary=? "
                "WHERE session_id=? AND notes_json=? AND summary IS ?",
                (notes, summary, sid, old_notes, old_summary),
            )
            if updated.rowcount == 0:
                deferred_sessions.append(sid)
        for row in pending:
            conn.execute("UPDATE suggestions SET status='rejected', reviewed_at=CURRENT_TIMESTAMP WHERE id=?", (row[0],))
        for sid in forgotten:
            for table in ("messages", "distilled", "consolidated_distillations", "sessions"):
                conn.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
    for row in pending:
        if row[4]:
            preview = root / ".suggestions" / row[4]
            if not preview.resolve().is_relative_to(root / ".suggestions"):
                raise ValueError("suggestion preview resolves outside its directory")
            preview.unlink(missing_ok=True)
    report = synchronize_recall_documents(conn, root, include_raw_transcripts=False, constitution_path=config.data['character']['constitution_path'])
    result.update(backup=str(backup), recall_deleted=report.deleted,
                  deferred_files=deferred_files, deferred_sessions=deferred_sessions)
    return result
