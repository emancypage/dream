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
from typing import NamedTuple

from backend import generate
from config import ConsolidateSettings, DreamConfig, load_config
from recall_query import query_terms, rank_lexical
from recall_types import RecallQuery
from recall_usage import memory_annotations, observation_window
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


class PatchEmptiesFile(PatchError):
    """A patch would leave nothing but frontmatter; deleting a file is a `remove`."""


def _without_frontmatter(text: str) -> str:
    """The text after a leading YAML frontmatter block, or the whole text."""
    if text.startswith("---\n"):
        end = text.find("\n---", 3)
        if end != -1:
            line_end = text.find("\n", end + 4)
            return "" if line_end == -1 else text[line_end + 1:]
    return text


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


_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _lf_offsets(text: str) -> tuple[str, list[int]]:
    """``text`` with CRLF and lone CR read as LF, and each character's raw offset (plus the end)."""
    chars: list[str] = []
    offsets: list[int] = []
    index = 0
    while index < len(text):
        offsets.append(index)
        if text[index] == "\r":
            chars.append("\n")
            index += 2 if text.startswith("\r\n", index) else 1
        else:
            chars.append(text[index])
            index += 1
    offsets.append(len(text))
    return "".join(chars), offsets


def patch_text(text: str, find: str, replace: str, *, allow_empty: bool = False) -> str:
    """Return text with the single NFC-normalized occurrence of find replaced.

    Pure: callers decide what to do with the result. The target must already be
    NFC so that matching never rewrites bytes outside the replaced span; zero or
    several matches are ambiguous and never applied as a best guess. A patch
    may not empty a file: that would bypass the `remove` grace period.

    Models see memory files with CRLF and lone CR read as LF, so in a file
    that has them the match is found on that view and mapped back to its raw
    span. Only that span is rewritten, and the replacement takes the line
    ending of the span (else the file's first one). Callers may pass raw or
    newline-translated text: both find the same match. LF-only text is
    matched and patched exactly as before.
    """
    if not find:
        raise PatchError("patch find text is empty")
    if unicodedata.normalize("NFC", text) != text:
        raise PatchError("target file is not NFC-normalized")
    needle = unicodedata.normalize("NFC", find)
    replacement = unicodedata.normalize("NFC", replace)
    if "\r" in text:
        shown, offsets = _lf_offsets(text)
        needle = _lf(needle)
    else:
        shown, offsets = text, None
    count = _occurrences(shown, needle)
    if count != 1:
        raise PatchError(f"find text matches {count} places")
    if not replace and not allow_empty:
        raise PatchError("empty replacement without a stated deletion in the rationale")
    if offsets is None:
        patched = text.replace(needle, replacement, 1)
        check = patched
    else:
        start = shown.find(needle)
        raw_start, raw_end = offsets[start], offsets[start + len(needle)]
        ending = _LINE_BREAK.search(text[raw_start:raw_end]) or _LINE_BREAK.search(text)
        patched = text[:raw_start] + _lf(replacement).replace("\n", ending.group(0)) + text[raw_end:]
        check = _lf(patched)
    if not _without_frontmatter(check).strip():
        raise PatchEmptiesFile("patch would empty the file; propose remove")
    return patched


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
    audit re-runs against the full store (every night in full mode, weekly through
    `dream audit` in subset mode), so a genuinely-closed topic is simply re-proposed
    once its mtime crosses the grace period.
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


class MemoryFile(NamedTuple):
    path: Path
    body: str
    sha256: str  # of the exact bytes `body` was decoded from


def memory_files(memory_root: Path, *, constitution_path) -> list[MemoryFile]:
    """Top-level memory files a model may see, sorted by name.

    Skips the constitution, hidden files and anything the local retention policy
    excludes, exactly as the full-mode memory store does. Each body is decoded
    (with read_text's newline handling) from the same bytes its SHA-256 covers,
    so a suggestion's base SHA always matches what the model was shown.
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
        data = f.read_bytes()
        body = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if policy.excludes(source_path=f.name, text=body) or policy.excludes_memory_path(memory_root, f.name):
            continue
        files.append(MemoryFile(f, body, hashlib.sha256(data).hexdigest()))
    return files


def read_current_memory(memory_root: Path, *, constitution_path) -> str:
    if not memory_root.exists():
        return "(memory root is empty — no files yet)"
    chunks = [render_memory_file(entry.path.name, entry.body)
              for entry in memory_files(memory_root, constitution_path=constitution_path)]
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


def _term_spans(finding: str) -> list[str]:
    """The finding itself, or consecutive whitespace-separated spans of it with at most 32 terms each.

    Spans keep the original words, so paths and other structured terms survive.
    """
    if len(query_terms(finding)) <= _FTS_QUERY_TERMS:
        return [finding]
    spans: list[str] = []
    current: list[str] = []
    terms: set[str] = set()
    for word in finding.split():
        word_terms = set(query_terms(word))
        if current and len(terms | word_terms) > _FTS_QUERY_TERMS:
            spans.append(" ".join(current))
            current, terms = [], set()
        current.append(word)
        terms |= word_terms
    if current:
        spans.append(" ".join(current))
    return spans


def _query_groups(findings: list[str]) -> list[str]:
    """Pack findings into query texts that rank_lexical reads completely.

    One query joining every finding would be cut to its first 32 terms, so the
    subset would reflect only the first finding or two of a batch. A finding
    with more terms than one query reads is split into spans first. Every
    finding of the batch lands in some group, and every group is ranked,
    because consolidation consumes the whole batch.
    """
    groups: list[str] = []
    current: list[str] = []
    terms: set[str] = set()
    for piece in (span for finding in findings for span in _term_spans(finding)):
        piece_terms = set(query_terms(piece))
        if not piece_terms:
            continue
        if current and len(terms | piece_terms) > _FTS_QUERY_TERMS:
            groups.append(" ".join(current))
            current, terms = [], set()
        current.append(piece)
        terms |= piece_terms
    if current:
        groups.append(" ".join(current))
    return groups


def _dense_for_subset(conn: sqlite3.Connection, query_text: str, embedder, eligible: set[str], *, limit: int = 200):
    """Return the dense candidate list for one subset query, or None.

    The dense list (approved memory under the embedder's current fingerprint)
    is passed to rank_lexical when an embedder is configured. Without one (the
    packaged default), or on any embedder or vector-store failure, subset
    selection stays lexical. Subset selection has no score threshold, so it
    needs no calibration row. Only documents of the ``eligible`` topic files
    (never MEMORY.md, which is always shown) take places under ``limit``, so
    vectors of files deleted or excluded since the last index sync cannot
    crowd out an eligible match.
    """
    if embedder is None:
        return None
    try:
        from recall_adapters import adapter_fingerprint, embed_query
        from recall_embeddings import dense_candidates, embedding_text, has_vectors
        fingerprint = adapter_fingerprint(embedder)
        if not has_vectors(conn, fingerprint, ("approved_memory",)):
            return []  # nothing embedded yet: no request to the server
        vector = embed_query(embedder, embedding_text(query_text))
        topics = eligible - {INDEX_FILE}

        def current_topics(ids):
            marks = ",".join("?" for _ in ids)
            rows = conn.execute(f"SELECT id, source_path FROM recall_documents WHERE id IN ({marks})", list(ids))
            return {document_id for document_id, source_path in rows if source_path in topics}

        return dense_candidates(conn, vector, fingerprint, ("approved_memory",), limit=limit, eligible=current_topics)
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
        # Only the files closest in meaning count as dense evidence, so a long
        # tail of weak matches cannot crowd out lexical ones.
        dense = _dense_for_subset(conn, text, embedder, eligible, limit=2 * limit)
        if dense is None:
            # Off, or failed: one failure is reported once, not once per group.
            embedder = None
        # Only approved memory files count, so only they may take places under the full-text limit.
        candidates = rank_lexical(conn, query, dense=dense, kinds=("approved_memory",))
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
    constitution_path,
    embedder=None,
) -> list[Path]:
    """Memory files to show a subset consolidation, MEMORY.md first.

    MEMORY.md, then at most ``settings.subset_files`` topic files ranked by the
    batch's findings, then every file whose slug (stem or frontmatter name)
    appears verbatim in a finding. A missing, stale or broken recall index falls
    back to MEMORY.md and the slug matches; it never fails the run.
    """
    files = memory_files(memory_root, constitution_path=constitution_path)
    by_name = {entry.path.name: entry for entry in files}
    selected: list[str] = [INDEX_FILE] if INDEX_FILE in by_name else []
    groups = _query_groups(_batch_findings(batch, _QUERY_FIELDS))
    if groups:
        try:
            selected += _ranked_memory_files(conn, groups, set(by_name), settings.subset_files, embedder)
        except Exception as exc:
            print(f"subset selection: recall index unavailable ({type(exc).__name__}); "
                  "using MEMORY.md and slug matches", file=sys.stderr)
    mention_text = "\n".join(_batch_findings(batch)).casefold()
    for entry in files:
        name = entry.path.name
        if name not in selected and any(_mentions(mention_text, slug) for slug in _slugs(entry.path, entry.body)):
            selected.append(name)
    return [by_name[name].path for name in selected]


def _other_memory_lines(files: list[MemoryFile], shown: list[str]) -> str:
    """Index lines (or bare names) of every eligible file the model is not shown."""
    hidden = [entry.path.name for entry in files if entry.path.name not in shown and entry.path.name != INDEX_FILE]
    index_body = next((entry.body for entry in files if entry.path.name == INDEX_FILE), "")
    lines: list[str] = []
    listed: set[str] = set()
    for line in index_body.splitlines():
        match = _INDEX_LINK.search(line)
        if match and match.group(1) in hidden and match.group(1) not in listed:
            lines.append(line)
            listed.add(match.group(1))
    lines += [f"- {name} (not in MEMORY.md)" for name in hidden if name not in listed]
    return "\n".join(lines) or "(none: every memory file is shown above)"


def _fit_whole_files(rendered: list[tuple[str, str]], limit: int) -> list[tuple[str, str]]:
    """The longest prefix of whole rendered files whose joined text stays within limit.

    A file is never cut mid-body. The first file that does not fit, and every
    file after it, is left out of the prompt and listed by its index line, so
    the host treats it as unseen and an update to it can only append. The
    selection puts MEMORY.md first.
    """
    kept: list[tuple[str, str]] = []
    size = 0
    for name, text in rendered:
        added = len(text) + (len("\n---\n") if kept else 0)
        if size + added > limit:
            break
        kept.append((name, text))
        size += added
    if len(kept) < len(rendered):
        print(
            f"WARNING: memory subset exceeds MAX_CURRENT_MEMORY_CHARS ({limit:,}); "
            f"{len(rendered) - len(kept)} of {len(rendered)} selected files left out "
            "and listed by their index lines instead.",
            file=sys.stderr,
        )
    return kept


def _drop(kind: str, target: str, reason: str) -> None:
    print(f"  dropped {kind} {target}: {reason}")


def _patch_mode_suggestions(
    payload: dict,
    memory_root: Path,
    *,
    allowed: tuple[str, ...],
    shown: set[str],
    files: list[MemoryFile],
    policy,
    projects: dict,
    subset: bool,
) -> list[Suggestion]:
    """Host checks for subset and audit output (full mode keeps its own loop).

    Only `allowed` kinds survive, and only for eligible memory files or new
    paths. Base SHAs come from the bytes the model was shown. `shown` holds only
    files whose whole text was in the prompt. A patch needs a nonempty `find`, a
    target shown in full, never targets MEMORY.md (index lines are only
    appended) and may not empty a file. In subset mode a `new` on an existing
    file becomes an `update`, and a full-body `update` of a file the model was
    not shown in full becomes an append to its current body.
    """
    root = memory_root.resolve()
    index = root / INDEX_FILE
    eligible = {entry.path.name: entry for entry in files}
    shown_paths = {(memory_root / name).resolve() for name in shown}
    out: list[Suggestion] = []
    for s in payload.get("suggestions", []):
        kind = s.get("kind", "new")
        target_path = s.get("target_path", "")
        body = s.get("body", "")
        resolved = (memory_root / target_path).resolve()
        # Look the target up by its normalized name, so "./topic.md" still counts as existing.
        state_key = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else target_path
        rationale = s.get("rationale", "")
        find = (s.get("find") or None) if kind == "patch" else None
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
            policy.excludes(source_path=target_path, text=body + "\n" + rationale + "\n" + (find or ""))
            or policy.excludes_memory_path(memory_root, target_path)
        ):
            continue
        entry = eligible.get(state_key)
        existed = entry is not None
        if not existed and resolved.exists():
            _drop(kind, target_path, "target exists but is not an eligible memory file")
            continue
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
            try:
                patch_text(entry.body, find, body, allow_empty=states_deletion(rationale))
            except PatchEmptiesFile as exc:
                _drop(kind, target_path, str(exc))
                continue
            except PatchError:
                pass  # kept: the preview explains it and accepting defers it
        if subset and kind == "new" and existed:
            kind = "update"
            rationale += f" (host: converted from `new` because {target_path} already exists)"
        if subset and kind == "update" and existed and resolved not in shown_paths:
            addition = _without_frontmatter(body).strip("\n")
            if not addition.strip():
                _drop(kind, target_path, "nothing to append to a file the model was not shown")
                continue
            body = entry.body.rstrip("\n") + "\n\n" + addition + "\n"
            rationale += " (host: appended: target not shown to the model)"
        out.append(Suggestion(
            kind=kind,
            target_path=target_path,
            body=body,
            rationale=rationale,
            source_sessions=s.get("source_sessions", []),
            base_sha256=entry.sha256 if entry else None,
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
    bodies = {entry.path.name: entry.body for entry in files}
    # Only files read above can be shown, so every shown body matches its base SHA.
    selected = [path.name for path in select_memory_subset(
        conn, memory_root, batch, settings, constitution_path=constitution_path, embedder=embedder)
        if path.name in bodies]
    usage = memory_annotations(conn, selected)
    fitted = _fit_whole_files(
        [(name, render_memory_file(name, bodies[name], annotation=usage[name])) for name in selected],
        MAX_CURRENT_MEMORY_CHARS,
    )
    # A file counts as shown only when the prompt holds all of it; the rest are
    # listed by index line, so the host treats them as unseen.
    shown = [name for name, _text in fitted]
    current_memory = "\n---\n".join(text for _name, text in fitted) or (
        "(no memory file fits the prompt budget; see the other memory files below)" if selected
        else "(memory root is empty — no files yet)"
    )
    prompt = (
        SUBSET_PROMPT_PATH.read_text(encoding="utf-8")
        .replace("{{today}}", dt.date.today().isoformat())
        .replace("{{memory_root}}", str(memory_root))
        .replace("{{session_count}}", str(len(batch)))
        .replace("{{store_count}}", str(len(files)))
        .replace("{{shown_count}}", str(len(shown)))
        .replace("{{usage_since}}", observation_window(conn))
        .replace("{{other_memory}}", _other_memory_lines(files, shown))
        .replace("{{current_memory}}", current_memory)
        .replace("{{distilled_batch}}", _serialize_batch(batch))
    )
    prompt += policy.context()
    res = generate("consolidate", prompt, CONSOLIDATE_PATCH_SCHEMA, model=model, config=cfg)
    suggestions = _patch_mode_suggestions(
        res.output, memory_root, allowed=("new", "update", "index", "patch"), shown=set(shown),
        files=files, policy=policy, projects=projects, subset=True,
    )
    suggestions = _drop_fresh_removals(suggestions, memory_root)
    _persist(conn, memory_root, suggestions, batch=batch, provider=res.provider, model=res.model)
    return suggestions


# ------------------------------------------------------------ weekly audit


# Room for the recall usage line run_audit adds under each file header.
_ANNOTATION_ALLOWANCE = len("last recalled: 2026-10-20; recalls: 999999\n")


def audit_chunks(files: list[MemoryFile], limit: int) -> list[list[MemoryFile]]:
    """Pack whole files, in order, into chunks whose rendered size stays within limit.

    Each file's size includes an allowance for its usage annotation. A file
    larger than the limit gets a chunk to itself and is never split.
    """
    chunks: list[list[MemoryFile]] = []
    current: list[MemoryFile] = []
    size = 0
    for entry in files:
        rendered = len(render_memory_file(entry.path.name, entry.body)) + _ANNOTATION_ALLOWANCE
        added = rendered + (len("\n---\n") if current else 0)
        if current and size + added > limit:
            chunks.append(current)
            current, size, added = [], 0, rendered
        current.append(entry)
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
    distillation is consumed. Each provider call gets its own consolidation run
    (`audit-<id>-chunk<N>`, with that call's provider and reported model) and its
    suggestions link to it. All runs are persisted together in one transaction,
    so a provider failure in any chunk persists nothing.
    """
    cfg = config or load_config()
    files = memory_files(memory_root, constitution_path=cfg.data["character"]["constitution_path"])
    topics = [entry for entry in files if entry.path.name != INDEX_FILE]
    if not topics:
        return []
    index_body = next((entry.body for entry in files if entry.path.name == INDEX_FILE), "(MEMORY.md does not exist)")
    policy = load_retention_policy(memory_root)
    chunks = audit_chunks(topics, MAX_CURRENT_MEMORY_CHARS // 4)
    # Usage is evidence the model may cite, never a reason to remove on its own;
    # _drop_fresh_removals and the review gate apply exactly as without it.
    usage = memory_annotations(conn, [entry.path.name for entry in topics])
    usage_since = observation_window(conn)
    template = AUDIT_PROMPT_PATH.read_text(encoding="utf-8")
    audit_run = f"audit-{uuid.uuid4().hex}"
    runs: list[tuple[str, str, str | None, list[Suggestion]]] = []
    for number, chunk in enumerate(chunks, 1):
        prompt = (
            template
            .replace("{{today}}", dt.date.today().isoformat())
            .replace("{{memory_root}}", str(memory_root))
            .replace("{{chunk_number}}", str(number))
            .replace("{{chunk_count}}", str(len(chunks)))
            .replace("{{chunk_file_count}}", str(len(chunk)))
            .replace("{{file_count}}", str(len(topics)))
            .replace("{{usage_since}}", usage_since)
            .replace("{{memory_index}}", index_body)
            .replace("{{current_memory}}", "\n---\n".join(
                render_memory_file(entry.path.name, entry.body, annotation=usage[entry.path.name])
                for entry in chunk))
        )
        prompt += policy.context()
        res = generate("audit", prompt, CONSOLIDATE_PATCH_SCHEMA, model=model, config=cfg)
        # audit_chunks never cuts a file, so the chunk's files are exactly what the
        # model read in full; anything else is not a valid target for this call.
        found = _patch_mode_suggestions(
            res.output, memory_root, allowed=("remove", "patch"), shown={entry.path.name for entry in chunk},
            files=files, policy=policy, projects={}, subset=False,
        )
        runs.append((f"{audit_run}-chunk{number}", res.provider, res.model, found))
    runs = [(run_id, provider, used_model, _drop_fresh_removals(found, memory_root))
            for run_id, provider, used_model, found in runs]
    _persist_runs(conn, memory_root, runs, batch=[])
    return [s for _run_id, _provider, _model, found in runs for s in found]


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
    _persist_runs(conn, memory_root, [(uuid.uuid4().hex, provider, model, suggestions)], batch=batch)


def _persist_runs(
    conn: sqlite3.Connection,
    memory_root: Path,
    runs: list[tuple[str, str, str | None, list[Suggestion]]],
    *,
    batch: list[tuple[str, str, dict, str]],
) -> None:
    """Persist provider runs and their suggestions in one transaction.

    `runs` holds one (run_id, provider, model, suggestions) entry per provider
    call, so each suggestion keeps the identity of the call that produced it.
    Previews are numbered across all runs. The batch's distillations are
    consumed by the first run. Any failure rolls back every run and removes the
    previews already written.
    """
    sug_dir = memory_root / ".suggestions"
    sug_dir.mkdir(parents=True, exist_ok=True)

    today = dt.date.today().isoformat()
    existing = list(sug_dir.glob(f"{today}-*.md"))
    start_n = len(existing) + 1

    written: list[Path] = []
    try:
        with conn:
            n = start_n
            for run_id, provider, model, suggestions in runs:
                n = _persist_run(conn, memory_root, sug_dir, today, n, written,
                                 run_id, provider, model, suggestions)
            for sid, _ts, _notes, dkey in batch:
                conn.execute(
                    "INSERT INTO consolidated_distillations(distillation_key, session_id, run_id) "
                    "VALUES (?, ?, ?)",
                    (dkey, sid, runs[0][0]),
                )
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise


def _persist_run(conn, memory_root: Path, sug_dir: Path, today: str, n: int, written: list[Path],
                 run_id: str, provider: str, model: str | None, suggestions: list[Suggestion]) -> int:
    """Insert one run row and its suggestions and previews; return the next preview number."""
    conn.execute(
        "INSERT INTO consolidation_runs(run_id, provider, model) VALUES (?, ?, ?)",
        (run_id, provider, model),
    )
    for s in suggestions:
        slug = re.sub(r"[^a-z0-9-]+", "-", s.target_path.lower().replace(".md", "")).strip("-")[:50] or "suggestion"
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
        # Exclusive create: a preview left by an earlier or concurrent run is never
        # overwritten and two rows never share a preview file.
        while True:
            fname = f"{today}-{n:03d}-{s.kind}-{slug}.md"
            fpath = sug_dir / fname
            try:
                handle = open(fpath, "x", encoding="utf-8")
            except FileExistsError:
                n += 1
                continue
            written.append(fpath)
            with handle:
                handle.write(header + preview)
            break
        n += 1
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
    return n
