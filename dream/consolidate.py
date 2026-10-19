"""
Cross-session consolidation through the configured model provider.
"""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from backend import generate
from config import ConsolidateSettings, DreamConfig, load_config
from recall_query import query_terms, rank_lexical
from recall_types import RecallQuery
from retention_policy import POLICY_FILE, load_retention_policy, filter_notes, empty_notes

DEFAULT_MODEL = None
PROMPT_PATH = Path(__file__).parent / "prompts" / "consolidate.md"
SUBSET_PROMPT_PATH = Path(__file__).parent / "prompts" / "consolidate-subset.md"
AUDIT_PROMPT_PATH = Path(__file__).parent / "prompts" / "audit.md"

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


# Subset and audit runs add the fact-level `patch` kind: `find` is the exact span to
# replace and `body` its replacement. Codex's --output-schema is strict, so `find` is
# required and empty for every other kind. Full mode keeps CONSOLIDATE_SCHEMA as is.
CONSOLIDATE_PATCH_SCHEMA = {
    "type": "object",
    "required": ["suggestions"],
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "target_path", "find", "body", "rationale", "source_sessions"],
                "additionalProperties": False,
                "properties": {
                    "kind":            {"type": "string", "enum": ["new", "update", "remove", "index", "patch"]},
                    "target_path":     {"type": "string"},
                    "find":            {"type": "string"},
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
    find: str | None = None


class PatchError(ValueError):
    """A `patch` cannot apply as proposed; the suggestion stays pending with this reason."""


_DELETION = re.compile(r"\b(?:delet(?:e|es|ed|ing|ion)|remov(?:e|es|ed|ing|al))\b", re.IGNORECASE)


def states_deletion(rationale: str | None) -> bool:
    """Whether a patch rationale says the span is deleted, which permits an empty replacement."""
    return bool(_DELETION.search(rationale or ""))


def _occurrences(text: str, needle: str) -> int:
    """Count occurrences of needle, overlapping ones included (each is a distinct match)."""
    count = 0
    start = text.find(needle)
    while start != -1:
        count += 1
        start = text.find(needle, start + 1)
    return count


def patch_text(text: str, find: str, replace: str, *, allow_empty: bool = False) -> str:
    """Return text with the single NFC-normalized occurrence of find replaced.

    Pure: callers decide what to do with the result. The target must already be
    NFC so that matching never rewrites bytes outside the replaced span; zero or
    several matches are ambiguous and never applied as a best guess.
    """
    if not find:
        raise PatchError("patch find text is empty")
    if unicodedata.normalize("NFC", text) != text:
        raise PatchError("target file is not NFC-normalized")
    needle = unicodedata.normalize("NFC", find)
    count = _occurrences(text, needle)
    if count != 1:
        raise PatchError(f"find text matches {count} places")
    if not replace and not allow_empty:
        raise PatchError("empty replacement without a stated deletion in the rationale")
    return text.replace(needle, unicodedata.normalize("NFC", replace), 1)


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


def render_memory_file(name: str, body: str, *, annotation: str | None = None) -> str:
    """Render one memory file for a model prompt.

    Every prompt that shows memory files (full, subset and audit) renders them
    here, so evidence about a file (an `annotation` line under its header) is
    added in one place. Without an annotation the output is the historical
    full-mode format.
    """
    header = f"### FILE: {name}\n" + (f"{annotation}\n" if annotation else "")
    return f"{header}\n{body}\n"


def memory_files(memory_root: Path, *, constitution_path) -> list[tuple[Path, str]]:
    """Top-level memory files a model may see, with their bodies, sorted by name.

    Skips the constitution, hidden files and anything the local retention policy
    excludes, exactly as the full-mode memory store does.
    """
    if not memory_root.exists():
        return []
    files = []
    policy = load_retention_policy(memory_root)
    from character.safety import is_constitution_source
    for f in sorted(memory_root.glob("*.md")):
        if is_constitution_source(f, constitution_path=constitution_path):
            continue
        if f.name.startswith("."):
            continue
        body = f.read_text(encoding='utf-8')
        if policy.excludes(source_path=f.name, text=body) or policy.excludes_memory_path(memory_root, f.name):
            continue
        files.append((f, body))
    return files


def read_current_memory(memory_root: Path, *, constitution_path) -> str:
    if not memory_root.exists():
        return "(memory root is empty — no files yet)"
    chunks = [render_memory_file(f.name, body) for f, body in memory_files(memory_root, constitution_path=constitution_path)]
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


# ------------------------------------------------------------ subset selection

INDEX_FILE = "MEMORY.md"
# The distilled findings that describe what a batch is about; every finding field
# (and the summary) is still scanned for file slugs.
_QUERY_FIELDS = ("preferences", "projects", "decisions", "user_facts")
# build_safe_fts_match reads only the first 32 terms of a query.
_FTS_QUERY_TERMS = 32
_INDEX_LINK = re.compile(r"\]\(([^)]+\.md)\)")
_FRONTMATTER_NAME = re.compile(r"^name:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", re.MULTILINE)


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _batch_findings(batch: list[tuple[str, str, dict, str]], fields=None) -> list[str]:
    findings: list[str] = []
    for _sid, _ts, notes, _dkey in batch:
        if not isinstance(notes, dict):
            continue
        for field in fields if fields is not None else notes:
            findings.extend(text for text in _strings(notes.get(field)) if text.strip())
    return findings


def _query_groups(findings: list[str]) -> list[str]:
    """Pack whole findings into query texts that rank_lexical reads completely.

    One query joining every finding would be cut to its first 32 terms, so the
    subset would reflect only the first finding or two of a batch.
    """
    groups: list[str] = []
    current: list[str] = []
    terms: set[str] = set()
    for finding in findings:
        finding_terms = set(query_terms(finding))
        if not finding_terms:
            continue
        if current and len(terms | finding_terms) > _FTS_QUERY_TERMS:
            groups.append(" ".join(current))
            current, terms = [], set()
        current.append(finding)
        terms |= finding_terms
    if current:
        groups.append(" ".join(current))
    return groups


def _dense_for_subset(conn: sqlite3.Connection, query_text: str, embedder):
    """Return the dense candidate list for one subset query, or None.

    The dense list is passed to rank_lexical when an embedder is configured.
    Without one (the packaged default), or on any embedder or vector-store
    failure, subset selection stays lexical.
    """
    if embedder is None:
        return None
    try:
        from recall_embeddings import dense_candidates
        fingerprint = embedder.fingerprint
        fingerprint = fingerprint() if callable(fingerprint) else fingerprint
        vector = embedder.embed([query_text])[0]
        return dense_candidates(conn, vector, fingerprint, ("approved_memory",))
    except Exception as exc:
        print(f"subset selection: dense candidates unavailable ({type(exc).__name__}); ranking lexically",
              file=sys.stderr)
        return None


def _ranked_memory_files(conn, groups: list[str], eligible: set[str], limit: int, embedder) -> list[str]:
    """Top approved-memory files for the batch queries, fused by reciprocal rank.

    recall_documents.source_path of an approved_memory document is its path
    relative to the memory root, so only names of eligible top-level files
    count; stale rows for deleted or excluded files are skipped.
    """
    scores: dict[str, float] = {}
    for text in groups:
        query = RecallQuery(text, None, "consolidate-subset", None, (), 0, frozenset(), False)
        dense = _dense_for_subset(conn, text, embedder)
        candidates = rank_lexical(conn, query) if dense is None else rank_lexical(conn, query, dense=dense)
        rank = 0
        for candidate in candidates:
            name = candidate.document.source_path
            if candidate.document.source_kind != "approved_memory" or name not in eligible or name == INDEX_FILE:
                continue
            rank += 1
            scores[name] = scores.get(name, 0.0) + 1.0 / (60 + rank)
    return sorted(scores, key=lambda name: (-scores[name], name))[:limit]


def _slugs(path: Path, body: str) -> set[str]:
    """A file's slugs: its stem and its frontmatter `name`."""
    slugs = {path.stem}
    if body.startswith("---\n"):
        end = body.find("\n---", 4)
        match = _FRONTMATTER_NAME.search(body[4:end if end != -1 else len(body)])
        if match:
            slugs.add(match.group(1).strip())
    return {slug.casefold() for slug in slugs if slug}


def _mentions(text: str, slug: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(slug) + r"(?![\w-])", text) is not None


def select_memory_subset(
    conn: sqlite3.Connection,
    memory_root: Path,
    batch: list[tuple[str, str, dict, str]],
    settings: ConsolidateSettings,
    *,
    embedder=None,
    constitution_path=None,
) -> list[Path]:
    """Memory files to show a subset consolidation, MEMORY.md first.

    MEMORY.md, then at most ``settings.subset_files`` topic files ranked by the
    batch's findings, then every file whose slug (stem or frontmatter name)
    appears verbatim in a finding. A missing, stale or broken recall index falls
    back to MEMORY.md and the slug matches; it never fails the run.
    """
    files = memory_files(memory_root, constitution_path=constitution_path)
    by_name = {path.name: (path, body) for path, body in files}
    selected: list[str] = [INDEX_FILE] if INDEX_FILE in by_name else []
    groups = _query_groups(_batch_findings(batch, _QUERY_FIELDS))
    if groups:
        try:
            selected += _ranked_memory_files(conn, groups, set(by_name), settings.subset_files, embedder)
        except Exception as exc:
            print(f"subset selection: recall index unavailable ({type(exc).__name__}); "
                  "using MEMORY.md and slug matches", file=sys.stderr)
    mention_text = "\n".join(_batch_findings(batch)).casefold()
    for path, body in files:
        if path.name not in selected and any(_mentions(mention_text, slug) for slug in _slugs(path, body)):
            selected.append(path.name)
    return [by_name[name][0] for name in selected]


def _other_memory_lines(files: list[tuple[Path, str]], shown: list[str]) -> str:
    """Index lines (or bare names) of every eligible file the model is not shown."""
    hidden = [path.name for path, _body in files if path.name not in shown and path.name != INDEX_FILE]
    index_body = next((body for path, body in files if path.name == INDEX_FILE), "")
    lines: list[str] = []
    listed: set[str] = set()
    for line in index_body.splitlines():
        match = _INDEX_LINK.search(line)
        if match and match.group(1) in hidden and match.group(1) not in listed:
            lines.append(line)
            listed.add(match.group(1))
    lines += [f"- {name} (not in MEMORY.md)" for name in hidden if name not in listed]
    return "\n".join(lines) or "(none: every memory file is shown above)"


def _bounded_memory_blob(chunks: list[str], what: str) -> str:
    blob = "\n---\n".join(chunks)
    if len(blob) > MAX_CURRENT_MEMORY_CHARS:
        print(
            f"WARNING: {what} ({len(blob):,} chars) exceeds "
            f"MAX_CURRENT_MEMORY_CHARS ({MAX_CURRENT_MEMORY_CHARS:,}) — truncating.",
            file=sys.stderr,
        )
        blob = blob[:MAX_CURRENT_MEMORY_CHARS] + "\n\n... [memory truncated for prompt budget] ..."
    return blob or "(memory root is empty — no files yet)"


def _drop(kind: str, target: str, reason: str) -> None:
    print(f"  dropped {kind} {target}: {reason}")


def _patch_mode_suggestions(
    payload: dict,
    memory_root: Path,
    *,
    allowed: tuple[str, ...],
    shown: set[str],
    base_state: dict[str, tuple[bool, str | None]],
    policy,
    projects: dict,
    subset: bool,
) -> list[Suggestion]:
    """Host checks for subset and audit output (full mode keeps its own loop).

    Only `allowed` kinds survive. A patch needs a nonempty `find`, an existing
    target the model was shown in full, and never targets MEMORY.md, whose index
    lines are only appended. In subset mode a `new` on an existing file becomes
    an `update` against that file's current SHA.
    """
    root = memory_root.resolve()
    index = root / INDEX_FILE
    shown_paths = {(memory_root / name).resolve() for name in shown}
    out: list[Suggestion] = []
    for s in payload.get("suggestions", []):
        kind = s.get("kind", "new")
        target_path = s.get("target_path", "")
        resolved = (memory_root / target_path).resolve()
        # Look the target up by its normalized name, so "./topic.md" still counts as existing.
        state_key = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else target_path
        rationale = s.get("rationale", "")
        if resolved == (memory_root / POLICY_FILE).resolve():
            continue
        if kind not in allowed:
            _drop(kind, target_path, "removals are left to the weekly audit" if subset and kind == "remove"
                  else "kind not allowed in this pass")
            continue
        if kind != "remove" and any(
            policy.excludes(project_slug=projects.get(sid)) for sid in s.get("source_sessions", [])
        ):
            continue
        if kind != "remove" and (
            policy.excludes(source_path=target_path, text=s.get("body", "") + "\n" + rationale)
            or policy.excludes_memory_path(memory_root, target_path)
        ):
            continue
        existed, base_hash = base_state.get(state_key, (False, None))
        find = (s.get("find") or None) if kind == "patch" else None
        if kind in ("patch", "remove") and resolved not in shown_paths:
            _drop(kind, target_path, "target was not shown in full")
            continue
        if kind == "patch":
            if resolved == index:
                _drop(kind, target_path, "MEMORY.md is never patched")
                continue
            if not find:
                _drop(kind, target_path, "patch has no find text")
                continue
            if not existed:
                _drop(kind, target_path, "patch target does not exist")
                continue
        if subset and kind == "new" and existed:
            kind = "update"
            rationale += f" (host: converted from `new` because {target_path} already exists)"
        if subset and kind == "update" and existed and resolved not in shown_paths:
            rationale += " (host: this file was not shown in full to the model; compare with its current body)"
        out.append(Suggestion(
            kind=kind,
            target_path=target_path,
            body=s.get("body", ""),
            rationale=rationale,
            source_sessions=s.get("source_sessions", []),
            base_sha256=base_hash,
            target_existed=existed,
            find=find,
        ))
    return out


def _consolidate_subset(
    conn: sqlite3.Connection,
    memory_root: Path,
    batch: list[tuple[str, str, dict, str]],
    *,
    policy,
    projects: dict,
    settings: ConsolidateSettings,
    model: str | None,
    cfg: DreamConfig,
    embedder,
) -> list[Suggestion]:
    # The staleness audit runs weekly in subset mode, so an empty batch has
    # nothing to merge and costs no provider call.
    if not batch:
        return []
    constitution_path = cfg.data["character"]["constitution_path"]
    files = memory_files(memory_root, constitution_path=constitution_path)
    bodies = {path.name: body for path, body in files}
    shown = [path.name for path in select_memory_subset(
        conn, memory_root, batch, settings, embedder=embedder, constitution_path=constitution_path)]
    base_state = _memory_state(memory_root)
    current_memory = _bounded_memory_blob(
        [render_memory_file(name, bodies[name]) for name in shown], "memory subset")
    prompt = (
        SUBSET_PROMPT_PATH.read_text(encoding="utf-8")
        .replace("{{today}}", dt.date.today().isoformat())
        .replace("{{memory_root}}", str(memory_root))
        .replace("{{session_count}}", str(len(batch)))
        .replace("{{store_count}}", str(len(files)))
        .replace("{{shown_count}}", str(len(shown)))
        .replace("{{other_memory}}", _other_memory_lines(files, shown))
        .replace("{{current_memory}}", current_memory)
        .replace("{{distilled_batch}}", _serialize_batch(batch))
    )
    prompt += policy.context()
    res = generate("consolidate", prompt, CONSOLIDATE_PATCH_SCHEMA, model=model, config=cfg)
    suggestions = _patch_mode_suggestions(
        res.output, memory_root, allowed=("new", "update", "index", "patch"), shown=set(shown),
        base_state=base_state, policy=policy, projects=projects, subset=True,
    )
    suggestions = _drop_fresh_removals(suggestions, memory_root)
    _persist(conn, memory_root, suggestions, batch=batch, provider=res.provider, model=res.model)
    return suggestions


# ------------------------------------------------------------ weekly audit


def audit_chunks(files: list[tuple[Path, str]], limit: int) -> list[list[tuple[Path, str]]]:
    """Pack whole files, in order, into chunks whose rendered size stays within limit.

    A file larger than the limit gets a chunk to itself and is never split.
    """
    chunks: list[list[tuple[Path, str]]] = []
    current: list[tuple[Path, str]] = []
    size = 0
    for path, body in files:
        rendered = len(render_memory_file(path.name, body))
        added = rendered + (len("\n---\n") if current else 0)
        if current and size + added > limit:
            chunks.append(current)
            current, size, added = [], 0, rendered
        current.append((path, body))
        size += added
    if current:
        chunks.append(current)
    return chunks


def run_audit(
    conn: sqlite3.Connection,
    memory_root: Path,
    *,
    model: str | None = DEFAULT_MODEL,
    config: DreamConfig | None = None,
) -> list[Suggestion]:
    """Weekly staleness audit over every topic file, in chunks; emits only remove and patch.

    MEMORY.md is shown with every chunk as context and is never a target. No
    distillation is consumed. Suggestions from all chunks are persisted together,
    so a provider failure in any chunk persists nothing.
    """
    cfg = config or load_config()
    files = memory_files(memory_root, constitution_path=cfg.data["character"]["constitution_path"])
    topics = [(path, body) for path, body in files if path.name != INDEX_FILE]
    if not topics:
        return []
    index_body = next((body for path, body in files if path.name == INDEX_FILE), "(MEMORY.md does not exist)")
    policy = load_retention_policy(memory_root)
    base_state = _memory_state(memory_root)
    chunks = audit_chunks(topics, MAX_CURRENT_MEMORY_CHARS // 4)
    template = AUDIT_PROMPT_PATH.read_text(encoding="utf-8")
    suggestions: list[Suggestion] = []
    provider = used_model = None
    for number, chunk in enumerate(chunks, 1):
        prompt = (
            template
            .replace("{{today}}", dt.date.today().isoformat())
            .replace("{{memory_root}}", str(memory_root))
            .replace("{{chunk_number}}", str(number))
            .replace("{{chunk_count}}", str(len(chunks)))
            .replace("{{chunk_file_count}}", str(len(chunk)))
            .replace("{{file_count}}", str(len(topics)))
            .replace("{{memory_index}}", index_body)
            .replace("{{current_memory}}", "\n---\n".join(render_memory_file(path.name, body) for path, body in chunk))
        )
        prompt += policy.context()
        res = generate("audit", prompt, CONSOLIDATE_PATCH_SCHEMA, model=model, config=cfg)
        provider, used_model = res.provider, res.model
        suggestions += _patch_mode_suggestions(
            res.output, memory_root, allowed=("remove", "patch"), shown={path.name for path, _ in chunk},
            base_state=base_state, policy=policy, projects={}, subset=False,
        )
    suggestions = _drop_fresh_removals(suggestions, memory_root)
    _persist(conn, memory_root, suggestions, batch=[], provider=provider, model=used_model)
    return suggestions


def _patch_preview(memory_root: Path, s: Suggestion) -> str:
    """Unified diff of a patch against the current file, or why it cannot apply."""
    try:
        current = (memory_root / s.target_path).read_text(encoding="utf-8")
        patched = patch_text(current, s.find or "", s.body, allow_empty=states_deletion(s.rationale))
    except (OSError, UnicodeDecodeError, PatchError) as exc:
        return f"(patch cannot apply as proposed: {exc})\n\nfind:\n{s.find}\n\nreplace with:\n{s.body}\n"
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"a/{s.target_path}",
        tofile=f"b/{s.target_path}",
        n=3,
    )
    return "".join(line if line.endswith("\n") else line + "\n" for line in diff)


def consolidate(
    conn: sqlite3.Connection,
    memory_root: Path,
    since_iso: str | None = None,
    model: str | None = DEFAULT_MODEL,
    max_sessions: int = 200,
    config: DreamConfig | None = None,
    *,
    mode: str | None = None,
    embedder=None,
) -> list[Suggestion]:
    cfg = config or load_config()
    settings = cfg.consolidate_settings()
    mode = mode or settings.mode
    if mode not in ("full", "subset"):
        raise ValueError(f"unknown consolidation mode: {mode!r}")
    batch = _fit_batch(collect_distilled_batch(conn, since_iso, limit=max_sessions))
    policy = load_retention_policy(memory_root)
    projects = dict(conn.execute("SELECT session_id, project_slug FROM sessions"))
    batch = [(sid, ts, empty_notes() if policy.excludes(project_slug=projects.get(sid))
              else filter_notes(notes, policy), dkey) for sid, ts, notes, dkey in batch]
    if mode == "subset":
        return _consolidate_subset(
            conn, memory_root, batch, policy=policy, projects=projects, settings=settings,
            model=model, cfg=cfg, embedder=embedder,
        )
    # Even with no new sessions, still run the staleness audit over the existing
    # store — that's the whole point of this pass. Only skip entirely when there is
    # truly nothing to look at.
    if not batch and not any(memory_root.glob("*.md")):
        return []

    base_state = _memory_state(memory_root)
    current_memory = read_current_memory(memory_root, constitution_path=cfg.data["character"]["constitution_path"])
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
    prompt += policy.context()
    res = generate("consolidate", prompt, CONSOLIDATE_SCHEMA, model=model, config=cfg)
    payload = res.output

    suggestions: list[Suggestion] = []
    for s in payload.get("suggestions", []):
        target_path = s.get("target_path", "")
        if (memory_root / target_path).resolve() == (memory_root / POLICY_FILE).resolve():
            continue
        if s.get("kind") != "remove" and any(
            policy.excludes(project_slug=projects.get(sid)) for sid in s.get("source_sessions", [])
        ):
            continue
        if s.get("kind") != "remove" and (
            policy.excludes(source_path=target_path, text=s.get("body", "") + "\n" + s.get("rationale", ""))
            or policy.excludes_memory_path(memory_root, target_path)
        ):
            continue
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
                preview = _patch_preview(memory_root, s) if s.kind == "patch" else (s.body or "(remove)")
                fpath.write_text(header + preview, encoding="utf-8")
                written.append(fpath)
                conn.execute(
                    "INSERT INTO suggestions(kind, target_path, body, rationale, source_sessions, "
                    "sug_file, base_sha256, target_existed, consolidation_run_id, find_text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        s.kind, s.target_path, s.body, s.rationale,
                        ",".join(s.source_sessions), fname, s.base_sha256,
                        int(s.target_existed), run_id, s.find if s.kind == "patch" else None,
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
