"""Safe lexical retrieval for canonical recall documents."""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import unicodedata
from pathlib import Path

from recall_scrub import scrub_text
from recall_types import LazyRecallCandidate, PendingExcerpt, RecallCandidate, RecallDocument, RecallQuery

_JIRA = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]+-\d+)(?![A-Za-z0-9])")
_WORD = re.compile(r"[\w]+", re.UNICODE)
_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could do does for from had has have how "
    "i if in into is it its me my of on or our please should that the their them "
    "there these they this those to us was we were what when where which who why "
    "will with would you your home tmp users user current status next step".split()
)
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


def document_path_text(project_slug: str | None, source_path: str, text: str) -> str:
    """The path-evidence haystack of one document, stored as ``recall_documents.path_text``.

    Tokens are separated by single spaces, so a padded substring test is a
    whole-token match.
    """
    return normalize_path_text(" ".join((project_slug or "", source_path, text)))


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


def query_terms(text: str) -> tuple[str, ...]:
    ignored = _STOP_WORDS | set(normalize_path_text(str(Path.home())).split())
    return tuple(dict.fromkeys(word for word in _WORD.findall(normalize_query_text(text)) if len(word) > 1 and word not in ignored))


def build_safe_fts_match(text: str) -> str | None:
    """Create a quoted FTS expression; user syntax never reaches MATCH."""
    words = query_terms(text)
    if not words:
        return None
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words[:32])


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


# Session starts match structure and paths over the FTS and dense candidates
# plus this many of the most recent eligible documents, never the whole store.
RECENT_POOL_SIZE = 500
_SQL_CHUNK = 500
_DOCUMENT_COLUMNS = (
    "id, content_sha256, source_kind, trust_level, project_slug, "
    "source_path, source_updated_at, indexed_at, source_version, text"
)


def _row_document(row) -> RecallDocument:
    return RecallDocument(
        id=row[0], content_sha256=row[1], source_kind=row[2], trust_level=row[3],
        project_slug=row[4], source_path=row[5], source_updated_at=row[6],
        indexed_at=row[7], source_version=row[8], text=row[9],
    )


def _document_columns(conn) -> set[str]:
    """The columns of ``recall_documents``; a missing table raises like any query on it."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(recall_documents)")}
    if not columns:
        raise sqlite3.OperationalError("no such table: recall_documents")
    return columns


def _load_pool(conn, ids, *, stored: bool) -> tuple[list[RecallDocument], dict[str, str]]:
    """Load the documents with these IDs and, when ``stored``, their path text, in chunks.

    The path text map holds only non-NULL values; a missing column (an
    unmigrated, read-only database) or a NULL value is computed on demand.
    """
    ids = list(ids)
    columns = _DOCUMENT_COLUMNS + (", path_text" if stored else "")
    documents: list[RecallDocument] = []
    path_texts: dict[str, str] = {}
    for start in range(0, len(ids), _SQL_CHUNK):
        chunk = ids[start:start + _SQL_CHUNK]
        rows = conn.execute(
            f"SELECT {columns} FROM recall_documents WHERE id IN ({','.join('?' for _ in chunk)})", chunk
        ).fetchall()
        for row in rows:
            documents.append(_row_document(row))
            if stored and row[10] is not None:
                path_texts[row[0]] = row[10]
    return documents, path_texts


def _load_documents(conn, ids: list[str] | None = None) -> list[RecallDocument]:
    """Load every document, or the documents with these IDs (unknown IDs are skipped)."""
    if ids is None:
        return [_row_document(row) for row in conn.execute(f"SELECT {_DOCUMENT_COLUMNS} FROM recall_documents")]
    return _load_pool(conn, ids, stored=False)[0]


def _eligible_ids(conn, ids, query: RecallQuery, excluded: set[str], kinds=None) -> set[str]:
    """The subset of ``ids`` that exist and pass the exclusions, the raw-transcript rule and ``kinds``."""
    ids = list(dict.fromkeys(ids))
    eligible: set[str] = set()
    for start in range(0, len(ids), _SQL_CHUNK):
        chunk = ids[start:start + _SQL_CHUNK]
        for doc_id, kind in conn.execute(
            f"SELECT id, source_kind FROM recall_documents WHERE id IN ({','.join('?' for _ in chunk)})", chunk
        ):
            if (doc_id not in excluded and (query.allow_raw_transcript or kind != "raw_transcript")
                    and (kinds is None or kind in kinds)):
                eligible.add(doc_id)
    return eligible


def _kinds_clause(kinds, column: str) -> tuple[str, tuple[str, ...]]:
    """``AND <column> IN (...)`` with its parameters, or nothing when every kind is allowed."""
    if kinds is None:
        return "", ()
    names = tuple(sorted(set(kinds)))
    return f" AND {column} IN ({','.join('?' for _ in names)})", names


def _recent_ids(conn, query: RecallQuery, excluded: set[str], limit: int, kinds=None) -> list[str]:
    """The ``limit`` most recent eligible document IDs by stored ``source_updated_at``.

    Writers store UTC ISO-8601 timestamps, whose text order is chronological
    to the second, so SQLite picks them without parsing every row. This only
    chooses the session-start pool; the recency list orders by parsed time.
    """
    allowed = "" if query.allow_raw_transcript else " WHERE source_kind != 'raw_transcript'"
    kind_clause, kind_params = _kinds_clause(kinds, "source_kind")
    if kind_clause and not allowed:
        allowed, kind_clause = " WHERE " + kind_clause[len(" AND "):], ""
    rows = conn.execute(
        f"SELECT id FROM recall_documents{allowed}{kind_clause} ORDER BY source_updated_at DESC, id LIMIT ?",
        (*kind_params, limit + len(excluded)),
    )
    return [doc_id for (doc_id,) in rows if doc_id not in excluded][:limit]


def rank_lexical(conn, query: RecallQuery, *, dense: list[tuple[str, float]] | None = None,
                 usage=None, kinds=None) -> list[RecallCandidate]:
    """Rank prompt topics by FTS; use fused project signals for session starts.

    ``kinds`` (None: every kind) is a hard source-kind filter bound into the
    full-text query and the recent pool, so documents of other kinds never
    take a place under their limits. Without it the queries are unchanged.

    ``dense`` is an optional pre-computed ``[(document_id, cosine)]`` list in
    rank order. Its eligible IDs (the same exclusions and raw-transcript rule)
    become a fifth RRF list named ``dense``; on the prompt path they join the
    FTS matches as topic candidates and the order becomes fused score first,
    because an FTS-first key would rank every dense-only candidate below every
    lexical hit. With no eligible dense IDs the result is exactly lexical.

    ``usage`` (``recall_usage.usage_map``, passed only with
    ``recall.usage_tie_break``) never adds or removes a candidate: a higher
    select count only breaks exact ties after trust, before best component rank.

    Structured and path evidence is matched over a bounded pool: on the
    prompt path the FTS and dense candidates, at session start those plus the
    ``RECENT_POOL_SIZE`` most recent eligible documents. A store with at most
    that many eligible documents therefore ranks exactly as an unbounded scan.
    A document is scrubbed only when a structured term occurs in its raw text
    or when its excerpt is first read (``LazyRecallCandidate``), at most once,
    and path matching reads the stored ``path_text`` when the column exists
    and is set.
    """
    stored_path_text = "path_text" in _document_columns(conn)
    structured = extract_structured_terms(query.query_text)
    is_prompt = not query.hook_event.startswith("session-start")
    allowed = "d.source_kind != 'raw_transcript'" if not query.allow_raw_transcript else "1=1"
    kind_clause, kind_params = _kinds_clause(kinds, "d.source_kind")
    excluded = set(query.excluded_source_ids)
    lists: dict[str, list[str]] = {}

    fts = build_safe_fts_match(query.query_text)
    if fts:
        try:
            rows = conn.execute(
                "SELECT d.id FROM recall_documents_fts f JOIN recall_documents d ON d.rowid=f.rowid "
                f"WHERE recall_documents_fts MATCH ? AND {allowed}{kind_clause} ORDER BY bm25(recall_documents_fts), d.id LIMIT 500",
                (fts, *kind_params),
            ).fetchall()
            lists["fts"] = [row[0] for row in rows if row[0] not in excluded]
        except Exception:
            lists["fts"] = []

    dense_pairs = list(dense or ())
    dense_eligible = _eligible_ids(conn, [doc_id for doc_id, _cosine in dense_pairs], query, excluded, kinds) if dense_pairs else set()
    dense_ids = [doc_id for doc_id, _cosine in dense_pairs if doc_id in dense_eligible]

    pool = dict.fromkeys(lists.get("fts", ()))
    pool.update(dict.fromkeys(dense_ids))
    if not is_prompt:
        pool.update(dict.fromkeys(_recent_ids(conn, query, excluded, RECENT_POOL_SIZE, kinds)))
    all_docs, path_texts = _load_pool(conn, pool, stored=stored_path_text)
    by_id = {doc.id: doc for doc in all_docs}
    # On a connection that is not immutable a writer can delete a row between
    # the queries above; such an ID is dropped rather than looked up.
    if "fts" in lists:
        lists["fts"] = [doc_id for doc_id in lists["fts"] if doc_id in by_id]
    dense_ids = [doc_id for doc_id in dense_ids if doc_id in by_id]
    home = Path.home()
    scrubbed: dict[str, str] = {}

    def scrub(text: str) -> str:
        return scrub_text(text, home)

    if structured:
        matches = []
        for doc in all_docs:
            # Scrubbing only replaces spans with "<...>" placeholders, which hold
            # no structured term, so a term missing from the raw text is missing
            # from the scrubbed text too; only possible matches are scrubbed.
            folded = doc.text.casefold()
            if not any(term in folded for term in structured):
                continue
            text = scrubbed.setdefault(doc.id, scrub(doc.text)).casefold()
            hits = sum(term in text for term in structured)
            if hits:
                matches.append((doc.id, -hits))
        matches.sort(key=lambda item: (item[1], item[0]))
        lists["structured"] = [item[0] for item in matches]

    path_query = normalize_path_text(" ".join((query.cwd or "", *query.repository_roots)))
    if path_query:
        path_tokens = [f" {token} " for token in query_terms(path_query)]
        matches = []
        if path_tokens:
            for doc in all_docs:
                haystack = path_texts.get(doc.id)
                if haystack is None:
                    haystack = document_path_text(doc.project_slug, doc.source_path, doc.text)
                haystack = f" {haystack} "
                overlap = sum(token in haystack for token in path_tokens)
                if overlap:
                    matches.append((doc.id, -overlap))
        matches.sort(key=lambda item: (item[1], item[0]))
        lists["paths"] = [item[0] for item in matches]

    if dense_ids:
        lists["dense"] = dense_ids

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
        best = min(components.values())
        excerpt = scrubbed.get(doc.id) or PendingExcerpt(doc.text, scrub)
        result.append(LazyRecallCandidate(doc, excerpt, dict(components), score, best))
    fts_first = is_prompt and not dense_ids
    result.sort(
        key=lambda candidate: (
            candidate.component_ranks.get("fts", float("inf")) if fts_first else 0,
            -candidate.score,
            trust_order.get(candidate.document.trust_level, 99),
            -usage.get(candidate.document.id, (0,))[0] if usage else 0,
            candidate.best_component_rank,
            -_parse_time(candidate.document.source_updated_at),
            candidate.document.id,
        )
    )
    return result
