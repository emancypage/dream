"""
Cross-session consolidation through the configured model provider.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from backend import generate
from config import DreamConfig, load_config

DEFAULT_MODEL = None
PROMPT_PATH = Path(__file__).parent / "prompts" / "consolidate.md"

MAX_DISTILLED_BATCH_CHARS = 200_000
# The live store measured 420,681 bytes on 2026-07-12 (75 files) — the old 80_000 cap
# silently truncated it to an alphabetical prefix, so most files (anything past early
# project_*) were never actually seen by the model, for this audit or the older
# "prefer updating to adding" dedup check. 800_000 covers today's store with headroom
# for growth. Provider-specific context constraints are handled by configuration/batching.
MAX_CURRENT_MEMORY_CHARS = 800_000

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "required": ["suggestions"],
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "target_path", "body", "rationale", "source_sessions"],
                "additionalProperties": False,
                "properties": {
                    "kind":            {"type": "string", "enum": ["new", "update", "remove", "index"]},
                    "target_path":     {"type": "string"},
                    "body":            {"type": "string"},
                    "rationale":       {"type": "string"},
                    "source_sessions": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


@dataclass
class Suggestion:
    kind: str
    target_path: str
    body: str
    rationale: str
    source_sessions: list[str]
    base_sha256: str | None = None
    target_existed: bool = False


REMOVE_GRACE_DAYS = 21


def _drop_fresh_removals(
    suggestions: list[Suggestion],
    memory_root: Path,
    grace_days: int = REMOVE_GRACE_DAYS,
) -> list[Suggestion]:
    """Filter out `remove` suggestions whose target file was touched too recently.

    Age is a filesystem fact, not something the model is shown (read_current_memory
    passes no mtimes, and LLMs are unreliable at date arithmetic) — so the "untouched
    for N days" half of the staleness criterion is enforced here, after the model
    responds, rather than asked of the model. A held-back suggestion isn't lost: the
    audit re-runs against the full store every night, so a genuinely-closed topic is
    simply re-proposed once its mtime crosses the grace period.
    """
    now = dt.datetime.now(dt.timezone.utc)
    kept: list[Suggestion] = []
    for s in suggestions:
        if s.kind != "remove":
            kept.append(s)
            continue
        tgt = memory_root / s.target_path
        if not tgt.exists():
            kept.append(s)
            continue
        age = now - dt.datetime.fromtimestamp(tgt.stat().st_mtime, tz=dt.timezone.utc)
        if age.days < grace_days:
            print(f"  held back: {s.target_path} — {grace_days - age.days}d remaining in grace period")
            continue
        kept.append(s)
    return kept


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def read_current_memory(memory_root: Path) -> str:
    if not memory_root.exists():
        return "(memory root is empty — no files yet)"
    chunks = []
    for f in sorted(memory_root.glob("*.md")):
        if f.name.startswith("."):
            continue
        chunks.append(f"### FILE: {f.name}\n\n{f.read_text(encoding='utf-8')}\n")
    blob = "\n---\n".join(chunks)
    if len(blob) > MAX_CURRENT_MEMORY_CHARS:
        print(
            f"WARNING: memory store ({len(blob):,} chars) exceeds "
            f"MAX_CURRENT_MEMORY_CHARS ({MAX_CURRENT_MEMORY_CHARS:,}) — truncating; "
            f"the staleness audit will not see every file this run.",
            file=sys.stderr,
        )
        blob = blob[:MAX_CURRENT_MEMORY_CHARS] + "\n\n... [memory truncated for prompt budget] ..."
    return blob or "(memory root is empty — no files yet)"


def collect_distilled_batch(
    conn: sqlite3.Connection,
    since_iso: str | None,
    limit: int | None = None,
) -> list[tuple[str, str, dict, str]]:
    """Return unconsolidated distillation identities, optionally time-filtered."""
    sql = """
        SELECT d.session_id,
               COALESCE(s.ended_at, s.started_at, d.distilled_at) as ts,
               d.notes_json,
               COALESCE(d.distillation_key, 'legacy:' || d.session_id) AS dkey
        FROM distilled d
        JOIN sessions s ON s.session_id = d.session_id
        LEFT JOIN consolidated_distillations cd
          ON cd.distillation_key=COALESCE(d.distillation_key, 'legacy:' || d.session_id)
        WHERE cd.distillation_key IS NULL
    """
    params: list = []
    if since_iso:
        sql += " AND COALESCE(s.ended_at, s.started_at, d.distilled_at) > ?"
        params.append(since_iso)
    sql += " ORDER BY ts ASC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    out = []
    for sid, ts, notes_json, dkey in conn.execute(sql, params).fetchall():
        try:
            notes = json.loads(notes_json)
        except json.JSONDecodeError:
            notes = {}
        out.append((sid, ts, notes, dkey))
    return out


def _serialize_batch(batch: list[tuple[str, str, dict, str]]) -> str:
    lines = []
    for sid, ts, notes, _dkey in batch:
        block = f"### session {sid} @ {ts}\n```json\n{json.dumps(notes, ensure_ascii=False, indent=2)}\n```\n"
        lines.append(block)
    return "\n".join(lines)


def _fit_batch(batch: list[tuple[str, str, dict, str]]) -> list[tuple[str, str, dict, str]]:
    """Select a whole-record prefix that fits; never silently truncate a selected row."""
    fitted = []
    total = 0
    for row in batch:
        block_len = len(_serialize_batch([row]))
        if fitted and total + block_len > MAX_DISTILLED_BATCH_CHARS:
            break
        fitted.append(row)
        total += block_len
    return fitted


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _memory_state(memory_root: Path) -> dict[str, tuple[bool, str | None]]:
    return {
        path.name: (True, _sha256(path))
        for path in memory_root.glob("*.md")
        if not path.name.startswith(".")
    }


def consolidate(
    conn: sqlite3.Connection,
    memory_root: Path,
    since_iso: str | None = None,
    model: str | None = DEFAULT_MODEL,
    max_sessions: int = 200,
    config: DreamConfig | None = None,
) -> list[Suggestion]:
    cfg = config or load_config()
    batch = _fit_batch(collect_distilled_batch(conn, since_iso, limit=max_sessions))
    # Even with no new sessions, still run the staleness audit over the existing
    # store — that's the whole point of this pass. Only skip entirely when there is
    # truly nothing to look at.
    if not batch and not any(memory_root.glob("*.md")):
        return []

    base_state = _memory_state(memory_root)
    current_memory = read_current_memory(memory_root)
    prompt = (
        _load_prompt()
        .replace("{{today}}", dt.date.today().isoformat())
        .replace("{{memory_root}}", str(memory_root))
        .replace("{{session_count}}", str(len(batch)))
        .replace("{{current_memory}}", current_memory)
        .replace("{{distilled_batch}}", _serialize_batch(batch))
    )

    # Consolidate is a single heavy call (whole distilled backlog + current memory →
    # structured suggestions). 600s was too tight once a multi-night backlog piled up
    # (timed out 06-07 + 06-08, batch only ~33k chars so it's the generation, not size).
    # 1800s gives headroom; systemd unit is TimeoutStartSec=infinity so this is the only cap.
    res = generate("consolidate", prompt, CONSOLIDATE_SCHEMA, model=model, config=cfg)
    payload = res.output

    suggestions: list[Suggestion] = []
    for s in payload.get("suggestions", []):
        target_path = s.get("target_path", "")
        existed, base_hash = base_state.get(target_path, (False, None))
        suggestions.append(Suggestion(
            kind=s.get("kind", "new"),
            target_path=target_path,
            body=s.get("body", ""),
            rationale=s.get("rationale", ""),
            source_sessions=s.get("source_sessions", []),
            base_sha256=base_hash,
            target_existed=existed,
        ))

    suggestions = _drop_fresh_removals(suggestions, memory_root)

    _persist(conn, memory_root, suggestions, batch=batch, provider=res.provider, model=res.model)
    return suggestions


def _persist(
    conn: sqlite3.Connection,
    memory_root: Path,
    suggestions: list[Suggestion],
    batch: list[tuple[str, str, dict, str]],
    provider: str,
    model: str | None,
) -> None:
    sug_dir = memory_root / ".suggestions"
    sug_dir.mkdir(parents=True, exist_ok=True)

    today = dt.date.today().isoformat()
    existing = list(sug_dir.glob(f"{today}-*.md"))
    start_n = len(existing) + 1

    run_id = uuid.uuid4().hex
    written: list[Path] = []
    try:
        with conn:
            conn.execute(
                "INSERT INTO consolidation_runs(run_id, provider, model) VALUES (?, ?, ?)",
                (run_id, provider, model),
            )
            for i, s in enumerate(suggestions, start=start_n):
                slug = re.sub(r"[^a-z0-9-]+", "-", s.target_path.lower().replace(".md", "")).strip("-")[:50] or "suggestion"
                fname = f"{today}-{i:03d}-{s.kind}-{slug}.md"
                fpath = sug_dir / fname
                header = (
                    f"<!-- dream suggestion\n"
                    f"kind: {s.kind}\n"
                    f"target_path: {s.target_path}\n"
                    f"rationale: {s.rationale}\n"
                    f"source_sessions: {', '.join(s.source_sessions)}\n"
                    f"base_sha256: {s.base_sha256 or ''}\n"
                    f"-->\n\n"
                )
                fpath.write_text(header + (s.body or "(remove)"), encoding="utf-8")
                written.append(fpath)
                conn.execute(
                    "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, "
                    "sug_file, base_sha256, target_existed, consolidation_run_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        s.kind, s.target_path, s.body, s.rationale,
                        ",".join(s.source_sessions), fname, s.base_sha256,
                        int(s.target_existed), run_id,
                    ),
                )
            for sid, _ts, _notes, dkey in batch:
                conn.execute(
                    "INSERT INTO consolidated_distillations(distillation_key, session_id, run_id) "
                    "VALUES (?, ?, ?)",
                    (dkey, sid, run_id),
                )
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
