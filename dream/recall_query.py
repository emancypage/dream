"""Safe lexical retrieval for canonical recall documents."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from pathlib import Path

from recall_scrub import scrub_text
from recall_types import RecallCandidate, RecallDocument, RecallQuery

_JIRA = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]+-\d+)(?![A-Za-z0-9])")
_WORD = re.compile(r"[\w]+", re.UNICODE)
_COMMANDS = frozenset(
    {
        "git", "gh", "codex", "dream", "pytest", "python", "python3", "pip", "uv",
        "npm", "npx", "pnpm", "yarn", "cargo", "go", "make", "docker", "kubectl",
        "terraform", "systemctl", "ssh", "curl", "rg", "grep", "find", "sed", "awk",
    }
)


def normalize_query_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = value.replace("\\", "/")
    return " ".join(value.split())


def normalize_path_text(text: str) -> str:
    value = normalize_query_text(text).replace("/", " ").replace("-", " ")
    parts = [part for part in value.split() if part not in {".", ".."}]
    return " ".join(parts)


def extract_structured_terms(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(text))
    terms: list[str] = []
    for match in _JIRA.finditer(normalized):
        term = match.group(1).casefold()
        if term not in terms:
            terms.append(term)
    for token in _WORD.findall(normalized.casefold()):
        if token in _COMMANDS and token not in terms:
            terms.append(token)
    return tuple(terms)


def build_safe_fts_match(text: str) -> str | None:
    """Create a quoted FTS expression; user syntax never reaches MATCH."""
    words = _WORD.findall(normalize_query_text(text))
    if not words:
        return None
    return " ".join('"' + word.replace('"', '""') + '"' for word in words[:32])


def _parse_time(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return 0.0


def _row_document(row) -> RecallDocument:
    return RecallDocument(
        id=row[0], content_sha256=row[1], source_kind=row[2], trust_level=row[3],
        project_slug=row[4], source_path=row[5], source_updated_at=row[6],
        indexed_at=row[7], source_version=row[8], text=row[9],
    )


def _load_documents(conn, ids: list[str] | None = None) -> list[RecallDocument]:
    sql = (
        "SELECT id, content_sha256, source_kind, trust_level, project_slug, "
        "source_path, source_updated_at, indexed_at, source_version, text "
        "FROM recall_documents"
    )
    params: list[str] = []
    if ids is not None:
        if not ids:
            return []
        sql += " WHERE id IN (" + ",".join("?" for _ in ids) + ")"
        params = ids
    return [_row_document(row) for row in conn.execute(sql, params).fetchall()]


def rank_lexical(conn, query: RecallQuery) -> list[RecallCandidate]:
    """Rank four independent lexical signals and fuse them with reciprocal rank."""
    normalized = normalize_query_text(query.query_text)
    structured = extract_structured_terms(query.query_text)
    allowed = "d.source_kind != 'raw_transcript'" if not query.allow_raw_transcript else "1=1"
    excluded = set(query.excluded_source_ids)
    all_docs = [d for d in _load_documents(conn) if d.id not in excluded and (query.allow_raw_transcript or d.source_kind != "raw_transcript")]
    by_id = {doc.id: doc for doc in all_docs}
    lists: dict[str, list[str]] = {}

    fts = build_safe_fts_match(query.query_text)
    if fts:
        try:
            rows = conn.execute(
                "SELECT d.id FROM recall_documents_fts f JOIN recall_documents d ON d.rowid=f.rowid "
                f"WHERE recall_documents_fts MATCH ? AND {allowed} ORDER BY bm25(recall_documents_fts), d.id LIMIT 500",
                (fts,),
            ).fetchall()
            lists["fts"] = [row[0] for row in rows if row[0] in by_id]
        except Exception:
            lists["fts"] = []

    if structured:
        matches = []
        for doc in all_docs:
            text = scrub_text(doc.text, Path.home()).casefold()
            hits = sum(term in text for term in structured)
            if hits:
                matches.append((doc.id, -hits))
        matches.sort(key=lambda item: (item[1], item[0]))
        lists["structured"] = [item[0] for item in matches]

    path_query = normalize_path_text(" ".join((query.cwd or "", *query.repository_roots)))
    if path_query:
        path_tokens = [token for token in path_query.split() if token not in {"home", "tmp", "users", "user"}]
        matches = []
        for doc in all_docs:
            haystack = normalize_path_text(" ".join((doc.project_slug or "", doc.source_path, doc.text)))
            overlap = sum(token in haystack.split() for token in path_tokens)
            if overlap:
                matches.append((doc.id, -overlap))
        matches.sort(key=lambda item: (item[1], item[0]))
        lists["paths"] = [item[0] for item in matches]

    # Recency is a supporting signal, not an independent source of unrelated
    # documents.  It boosts documents found by content, structure, or path
    # evidence and therefore cannot inject a recent secret into every query.
    eligible_ids = {doc_id for values in lists.values() for doc_id in values}
    if eligible_ids:
        lists["recency"] = [
            doc.id for doc in sorted(
                [doc for doc in all_docs if doc.id in eligible_ids],
                key=lambda doc: (-_parse_time(doc.source_updated_at), doc.id),
            )
        ]

    active = [values for values in lists.values() if values]
    if not active:
        return []
    ranks: dict[str, dict[str, int]] = {}
    for name, values in lists.items():
        for rank, doc_id in enumerate(values, 1):
            ranks.setdefault(doc_id, {})[name] = rank
    count = len(active)
    trust_order = {"user_approved": 0, "model_distilled": 1, "untrusted_transcript": 2}
    result: list[RecallCandidate] = []
    for doc_id, components in ranks.items():
        doc = by_id[doc_id]
        score = sum(1.0 / (60 + rank) for rank in components.values()) / count
        excerpt = scrub_text(doc.text, Path.home())
        best = min(components.values())
        result.append(RecallCandidate(doc, excerpt, dict(components), score, best))
    result.sort(
        key=lambda candidate: (
            -candidate.score,
            trust_order.get(candidate.document.trust_level, 99),
            candidate.best_component_rank,
            -_parse_time(candidate.document.source_updated_at),
            candidate.document.id,
        )
    )
    return result
