"""
Per-session distillation through the provider configured for the distill stage.
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend import generate
from config import DreamConfig, load_config
from retention_policy import load_retention_policy, filter_notes

DEFAULT_MODEL = None
PROMPT_PATH = Path(__file__).parent / "prompts" / "distill.md"

# Heuristic upper bound on transcript chars sent per call.
MAX_TRANSCRIPT_CHARS = 150_000
HEAD_TAIL_RATIO = 0.6

# Provider output schema. Empty arrays are valid for trivial sessions.
DISTILL_SCHEMA = {
    "type": "object",
    "required": ["summary", "user_facts", "preferences", "projects", "references", "decisions", "open_threads"],
    "additionalProperties": False,
    "properties": {
        "summary":      {"type": "string"},
        "user_facts":   {"type": "array", "items": {"type": "string"}},
        "preferences":  {"type": "array", "items": {"type": "string"}},
        "projects":     {"type": "array", "items": {"type": "string"}},
        "references":   {"type": "array", "items": {"type": "string"}},
        "decisions":    {"type": "array", "items": {"type": "string"}},
        "open_threads": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class DistillResult:
    session_id: str
    notes: dict
    summary: str
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_tokens: int | None
    cache_read_tokens: int | None
    cost_usd: float | None
    provider: str
    model: str | None
    duration_ms: int


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_version() -> str:
    return hashlib.sha256(_load_prompt().encode("utf-8")).hexdigest()


def _render_transcript(messages: Iterable[tuple[str, str, str | None]]) -> str:
    lines = []
    for role, text, ts in messages:
        prefix = "USER" if role == "user" else "ASSISTANT"
        lines.append(f"--- {prefix} [{ts}] ---" if ts else f"--- {prefix} ---")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _truncate(transcript: str) -> str:
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript
    head_n = int(MAX_TRANSCRIPT_CHARS * HEAD_TAIL_RATIO)
    tail_n = MAX_TRANSCRIPT_CHARS - head_n
    return (
        transcript[:head_n]
        + f"\n\n... [TRUNCATED {len(transcript) - MAX_TRANSCRIPT_CHARS} chars in middle] ...\n\n"
        + transcript[-tail_n:]
    )


def load_session_messages(conn: sqlite3.Connection, session_id: str) -> list[tuple[str, str, str | None]]:
    rows = conn.execute(
        "SELECT role, text, timestamp FROM messages WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def distill_session(
    conn: sqlite3.Connection,
    session_id: str,
    model: str | None = DEFAULT_MODEL,
    config: DreamConfig | None = None,
    memory_root: Path | None = None,
) -> DistillResult | None:
    session_row = conn.execute(
        "SELECT started_at, cwd, project_slug, source_revision, parser_version "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not session_row:
        return None
    started_at, cwd, project_slug, source_revision, parser_version = session_row

    cfg = config or load_config()
    policy = load_retention_policy(memory_root or cfg.memory_root)
    if policy.excludes(project_slug=project_slug):
        return None

    messages = load_session_messages(conn, session_id)
    if not messages:
        return None

    transcript = _truncate(_render_transcript(messages))
    prompt = (
        _load_prompt()
        .replace("{{started_at}}", started_at or "unknown")
        .replace("{{cwd}}", cwd or "unknown")
        .replace("{{project_slug}}", project_slug or "unknown")
        .replace("{{transcript}}", transcript)
    )

    prompt += policy.context()
    res = generate("distill", prompt, DISTILL_SCHEMA, model=model, config=cfg)
    notes = filter_notes(res.output, policy)
    summary = notes.get("summary", "")[:500]

    result = DistillResult(
        session_id=session_id,
        notes=notes,
        summary=summary,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        cache_creation_tokens=res.cache_creation_tokens,
        cache_read_tokens=res.cache_read_tokens,
        cost_usd=res.total_cost_usd,
        provider=res.provider,
        model=res.model,
        duration_ms=res.duration_ms,
    )

    key_material = "\0".join([
        session_id,
        source_revision or "",
        parser_version or "",
        prompt_version(),
        res.provider,
        res.model or "",
        json.dumps(cfg.stage("distill"), sort_keys=True),
    ])
    distillation_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    usage_json = json.dumps({
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "cache_creation_tokens": res.cache_creation_tokens,
        "cache_read_tokens": res.cache_read_tokens,
    }) if res.usage is not None else None

    conn.execute(
        """
        INSERT INTO distilled(
            session_id, distillation_key, provider, model, source_revision,
            parser_version, prompt_version, provider_options, input_tokens,
            output_tokens, usage_json, duration_ms, notes_json, summary, distilled_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET
            distillation_key=excluded.distillation_key,
            provider=excluded.provider,
            model=excluded.model,
            source_revision=excluded.source_revision,
            parser_version=excluded.parser_version,
            prompt_version=excluded.prompt_version,
            provider_options=excluded.provider_options,
            input_tokens=excluded.input_tokens,
            output_tokens=excluded.output_tokens,
            usage_json=excluded.usage_json,
            duration_ms=excluded.duration_ms,
            notes_json=excluded.notes_json,
            summary=excluded.summary,
            distilled_at=CURRENT_TIMESTAMP
        """,
        (
            session_id, distillation_key, res.provider, res.model, source_revision,
            parser_version, prompt_version(), json.dumps(cfg.stage("distill"), sort_keys=True),
            res.input_tokens, res.output_tokens, usage_json, res.duration_ms,
            json.dumps(notes, ensure_ascii=False), summary,
        ),
    )
    conn.commit()
    if memory_root is not None:
        from recall_documents import synchronize_recall_documents

        try:
            synchronize_recall_documents(
                conn,
                memory_root,
                include_raw_transcripts=False,
                constitution_path=cfg.data["character"]["constitution_path"],
            )
        except Exception:
            # Synchronization is maintenance, never fatal to a successful
            # distillation: the distilled row above is already committed.
            pass
    return result


def _drop_unchanged_local_routes(conn: sqlite3.Connection, rows: list, stage: dict) -> list:
    """Refresh filter for an openai-compatible distill route.

    Such rows record provider "openai-compatible" and the server-reported
    model, which never equal the stage's provider name and model, so the
    column comparison flags every session. Compare the recorded stage settings
    (provider_options) with the current stage instead. Rows without
    provider_options keep the column comparison. A model changed only inside
    [providers.<name>] is not detected.
    """
    recorded = {
        row[0]: (row[1], row[2])
        for row in conn.execute(
            """
            SELECT d.session_id, d.provider_options,
                   COALESCE(d.source_revision, '') <> COALESCE(s.source_revision, '')
                OR COALESCE(d.parser_version, '') <> COALESCE(s.parser_version, '')
                OR COALESCE(d.prompt_version, '') <> ?
            FROM distilled d JOIN sessions s ON s.session_id = d.session_id
            """,
            (prompt_version(),),
        )
    }
    kept = []
    for row in rows:
        options, content_changed = recorded.get(row[0], (None, True))
        if options is not None and not content_changed:
            try:
                previous = json.loads(options)
            except ValueError:
                previous = None
            if (isinstance(previous, dict) and previous.get("provider") == stage["provider"]
                    and previous.get("model") == stage.get("model")):
                continue
        kept.append(row)
    return kept


def sessions_needing_distill(
    conn: sqlite3.Connection,
    min_chars: int = 500,
    project_slug: str | None = None,
    config: DreamConfig | None = None,
    refresh_config: bool = False,
) -> list[tuple[str, int]]:
    cfg = config or load_config()
    stage = cfg.stage("distill")
    provider_name = stage["provider"]
    model_name = stage.get("model")
    sql = """
        SELECT s.session_id, s.total_chars
        FROM sessions s
        LEFT JOIN distilled d ON d.session_id = s.session_id
        WHERE (
              d.session_id IS NULL
           OR COALESCE(d.source_revision, '') <> COALESCE(s.source_revision, '')
           OR COALESCE(d.parser_version, '') <> COALESCE(s.parser_version, '')
           OR (? AND (
                  COALESCE(d.prompt_version, '') <> ?
               OR COALESCE(d.provider, '') <> ?
               OR COALESCE(d.model, '') <> COALESCE(?, '')
           ))
        )
          AND s.total_chars >= ?
    """
    params: list = [int(refresh_config), prompt_version(), provider_name, model_name, min_chars]
    if project_slug:
        sql += " AND s.project_slug = ?"
        params.append(project_slug)
    sql += " ORDER BY s.started_at DESC"
    policy = load_retention_policy(cfg.memory_root)
    excluded_sessions = {row[0] for row in conn.execute("SELECT session_id, project_slug FROM sessions")
                         if policy.excludes(project_slug=row[1])}
    rows = conn.execute(sql, params).fetchall()
    if refresh_config and cfg.provider(provider_name).get("type") == "openai-compatible":
        rows = _drop_unchanged_local_routes(conn, rows, stage)
    return [(r[0], r[1]) for r in rows if r[0] not in excluded_sessions]
