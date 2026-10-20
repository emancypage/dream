"""Fixture-based evaluation and held-out threshold calibration."""

from __future__ import annotations

import bisect
import hashlib
import json
import statistics
import time
import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

from recall_adapters import adapter_fingerprint
from recall_context import EMBEDDER_MODE, retrieve, with_adapters
from recall_documents import embed_recall_documents
from recall_query import rank_lexical
from recall_render import render_context
from recall_select import embedder_calibration_version, select_recall_candidates
from recall_types import CalibrationRecord, RecallQuery, RecallSettings

# Every fixture under the repository's fixtures directory is public by definition.
FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
UNTAGGED_LANGUAGE = "untagged"
ALL_LANGUAGES = "all"


@dataclass(frozen=True)
class SliceMetrics:
    """Headline retrieval metrics for one (mode, language) slice of a fixture."""

    recall_at_1: float
    recall_at_5: float
    mrr: float
    forbidden_count: int
    p95_latency_ms: float
    query_count: int
    fallback_count: int


@dataclass(frozen=True)
class EvaluationReport:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    injected_result_precision: float
    forbidden_count: int
    empty_result_correctness: float
    duplicate_count: int
    codepoints: int
    p50_latency_ms: float
    p95_latency_ms: float
    calibration_version: str | None
    query_count: int
    # {mode: {language: SliceMetrics}}; kept out of to_dict() so the default
    # report stays byte-for-byte what it was before the breakdown existed.
    breakdown: dict = field(default_factory=dict, compare=False)

    @property
    def metrics(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("breakdown")
        return data

    def breakdown_dict(self) -> dict:
        return {
            mode: {language: asdict(values) for language, values in sorted(languages.items())}
            for mode, languages in self.breakdown.items()
        }


def _canonical(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _default_settings() -> RecallSettings:
    from recall_types import AdapterSettings
    return RecallSettings(True, False, False, True, 6000, 4000, 1800, 1200, "", "", AdapterSettings(False, "none", False), AdapterSettings(False, "none", False))


def _fixture(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list) or not isinstance(data.get("queries"), list):
        raise ValueError("fixture must contain documents and queries arrays")
    return data


def _ensure_documents(conn, documents: list[dict]) -> None:
    for document in documents:
        text = str(document.get("text", ""))
        content_hash = document.get("content_sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO recall_documents(id, content_sha256, source_kind, trust_level, project_slug, source_path, source_updated_at, indexed_at, source_version, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET content_sha256=excluded.content_sha256, source_kind=excluded.source_kind, trust_level=excluded.trust_level, project_slug=excluded.project_slug, source_path=excluded.source_path, source_updated_at=excluded.source_updated_at, source_version=excluded.source_version, text=excluded.text",
            (document["id"], content_hash, document["source_kind"], document["trust_level"], document.get("project_slug"), document.get("source_path", document["id"]), document.get("source_updated_at", ""), document.get("indexed_at", document.get("source_updated_at", "")), document.get("source_version", "v1"), text),
        )
    conn.commit()
    if documents:
        from dream import rebuild_recall_fts
        rebuild_recall_fts(conn)


def _query(item: dict, settings: RecallSettings) -> RecallQuery:
    event = str(item.get("event", "prompt"))
    return RecallQuery(
        query_text=str(item.get("query", "")),
        session_id=item.get("session_id"),
        hook_event=event,
        cwd=item.get("cwd"),
        repository_roots=tuple(item.get("repository_roots", ())),
        requested_codepoint_budget=settings.session_start_budget_codepoints if event.startswith("session-start") else settings.prompt_budget_codepoints,
        excluded_source_ids=frozenset(item.get("excluded", ())),
        allow_raw_transcript=bool(item.get("allow_raw_transcript", False)) and settings.allow_raw_transcript_prompt,
    )


def _relevant(item: dict) -> set:
    # Public fixtures say "relevant"; generated private fixtures say "expected_ids".
    return set(item["relevant"] if "relevant" in item else item.get("expected_ids", ()))


def _p95(latencies: list[float]) -> float:
    return sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0


def _run_queries(data: dict, conn, settings: RecallSettings, calibration: dict, embedder=None) -> list[dict]:
    """Run every fixture query once and keep per-query evidence for aggregation.

    Queries go through the same ``retrieve`` pipeline as the recall hooks,
    read-only: the dense list, the rescore and the per-mode threshold.
    """
    outcomes: list[dict] = []
    for item in data["queries"]:
        language = str(item.get("language") or UNTAGGED_LANGUAGE)
        if language == ALL_LANGUAGES:
            raise ValueError(f"fixture query language {ALL_LANGUAGES!r} is reserved for the aggregate slice")
        query = _query(item, settings)
        started = time.perf_counter()
        outcome = retrieve(conn, query, settings, calibration, embedder=embedder, allow_cache_writes=False,
                           require_calibration=False)
        fallback = outcome.fallback_reason
        selected = outcome.selected
        rendered = render_context(selected, query.requested_codepoint_budget, query.query_text)
        latency = (time.perf_counter() - started) * 1000
        selected_ids = [candidate.document.id for candidate in selected]
        relevant = _relevant(item)
        outcomes.append({
            "language": language,
            "latency": latency,
            "selected_ids": selected_ids,
            "relevant": relevant,
            "ranks": [index + 1 for index, doc_id in enumerate(selected_ids) if doc_id in relevant],
            "forbidden": len(set(item.get("forbidden", ())).intersection(selected_ids)),
            "duplicates": len(selected_ids) - len({candidate.document.content_sha256 for candidate in selected}),
            "codepoints": len(rendered),
            "fallback": fallback is not None,
        })
    return outcomes


def _ranked_metrics(outcomes: list[dict]) -> tuple[float, float, float, float]:
    rows = [outcome for outcome in outcomes if outcome["relevant"]]
    def at(k: int) -> float:
        return sum(1 for row in rows if any(rank <= k for rank in row["ranks"])) / len(rows) if rows else 0.0
    mrr = sum(1 / min(row["ranks"]) for row in rows if row["ranks"]) / len(rows) if rows else 0.0
    return at(1), at(3), at(5), mrr


def _slice(outcomes: list[dict]) -> SliceMetrics:
    recall_at_1, _recall_at_3, recall_at_5, mrr = _ranked_metrics(outcomes)
    return SliceMetrics(
        recall_at_1, recall_at_5, mrr,
        sum(outcome["forbidden"] for outcome in outcomes),
        _p95([outcome["latency"] for outcome in outcomes]),
        len(outcomes),
        sum(1 for outcome in outcomes if outcome["fallback"]),
    )


def _breakdown(outcomes: list[dict]) -> dict[str, SliceMetrics]:
    languages = sorted({outcome["language"] for outcome in outcomes})
    result = {ALL_LANGUAGES: _slice(outcomes)}
    for language in languages:
        result[language] = _slice([outcome for outcome in outcomes if outcome["language"] == language])
    return result


def _lexical_settings(settings: RecallSettings) -> RecallSettings:
    # The lexical baseline never consults an adapter, whatever the configuration says.
    return with_adapters(settings, embedder=False, reranker=False)


def _embedder_settings(settings: RecallSettings) -> RecallSettings:
    # Measure the embedder alone: force it on and keep the reranker out of the mode.
    return with_adapters(settings, embedder=True, reranker=False)


def _embed_fixture_documents(conn, embedder, settings: RecallSettings):
    """Embed fixture documents into the throwaway eval database.

    Read-only recall never embeds documents, so without this step the
    embedder mode would silently equal lexical. A failure is not fatal: the
    queries then report it as a fallback.
    """
    try:
        return embed_recall_documents(conn, embedder, include_raw_transcripts=settings.allow_raw_transcript_prompt)
    except Exception:
        return None


def _evaluate(data: dict, conn, settings: RecallSettings, threshold: float | None = None, *, embedder=None,
              primary_mode: str = "lexical") -> EvaluationReport:
    """Run every mode; ``threshold`` applies to ``primary_mode``, which the top-level fields describe."""
    _ensure_documents(conn, data["documents"])
    mode = "lexical"

    def calibration(name: str) -> dict:
        return {name: {"threshold": threshold}} if threshold is not None and name == primary_mode else {}

    outcomes = {mode: _run_queries(data, conn, _lexical_settings(settings), calibration(mode))}
    if embedder is not None:
        _embed_fixture_documents(conn, embedder, settings)
        outcomes[EMBEDDER_MODE] = _run_queries(data, conn, _embedder_settings(settings), calibration(EMBEDDER_MODE), embedder=embedder)
    lexical = outcomes[primary_mode]
    latencies = [outcome["latency"] for outcome in lexical]
    all_forbidden = sum(outcome["forbidden"] for outcome in lexical)
    duplicate_count = sum(outcome["duplicates"] for outcome in lexical)
    total_codepoints = sum(outcome["codepoints"] for outcome in lexical)
    empty_correct = [1.0 if not outcome["selected_ids"] else 0.0 for outcome in lexical if not outcome["relevant"]]
    query_count = len(data["queries"])
    recall_at_1, recall_at_3, recall_at_5, mrr = _ranked_metrics(lexical)
    injected = [row for row in data["queries"] if row.get("injected")]
    injected_hits = sum(1 for row in injected if set(row.get("injected", ())).intersection({candidate.document.id for candidate in rank_lexical(conn, _query(row, settings))}))
    injected_precision = injected_hits / len(injected) if injected else 1.0
    return EvaluationReport(
        recall_at_1, recall_at_3, recall_at_5, mrr, injected_precision, all_forbidden,
        statistics.mean(empty_correct) if empty_correct else 1.0,
        duplicate_count, total_codepoints,
        statistics.median(latencies) if latencies else 0.0,
        _p95(latencies),
        None, query_count,
        {name: _breakdown(values) for name, values in outcomes.items()},
    )


def evaluate_fixture_file(path, conn, settings=None, *, embedder=None) -> EvaluationReport:
    """Evaluate a fixture against the supplied database.

    The report always carries a per-mode, per-language breakdown: ``lexical``,
    plus ``lexical_plus_embedder`` when an embedder is supplied.
    """
    return _evaluate(_fixture(Path(path)), conn, settings or _default_settings(), embedder=embedder)


def _score(record: EvaluationReport) -> tuple[float, float]:
    return record.mrr, record.recall_at_3


def _is_public_fixture(path: Path) -> bool:
    if path.name.lower() == "public.json" or "public" in {part.lower() for part in path.parts}:
        return True
    return path.expanduser().resolve().is_relative_to(FIXTURES_ROOT)


@dataclass(frozen=True)
class _Cut:
    """One query's evidence for every threshold in ``(previous score, score]``."""

    best_rank: int | None
    precision: float
    forbidden: int


def _query_cuts(item: dict, query: RecallQuery, outcome, settings: RecallSettings, mode: str) -> tuple[list[float], list[_Cut]]:
    """Selection evidence of one retrieved query for each distinct score cut.

    A threshold ``t`` keeps the candidates scoring at least ``t``, so the
    query has one selection per distinct candidate score plus "nothing
    passes". Only selection reruns here; ranking ran once.
    """
    cuts = sorted({candidate.score for candidate in outcome.candidates})
    relevant = _relevant(item)
    forbidden = set(item.get("forbidden", ()))
    rows: list[_Cut] = []
    for value in [*cuts, float("inf")]:
        selected = select_recall_candidates(outcome.candidates, query, settings, {mode: {"threshold": value}}, mode=outcome.mode)
        ids = [candidate.document.id for candidate in selected]
        ranks = [index + 1 for index, doc_id in enumerate(ids) if doc_id in relevant]
        rows.append(_Cut(min(ranks) if ranks else None,
                         len(relevant.intersection(ids)) / len(ids) if ids else 0.0,
                         len(forbidden.intersection(ids))))
    return cuts, rows


def calibration_key(rows: list[tuple[bool, _Cut]], threshold: float) -> tuple[float, float, float, float]:
    """Order thresholds: MRR, then Recall@3, then precision, then the lower threshold.

    ``rows`` holds ``(has_relevant, cut)`` per query. Precision is the mean
    share of relevant documents among the selected ones (0 when nothing is
    selected) over queries with relevant documents; without it every tie went
    to the lowest threshold, which selects everything. Any forbidden result
    disqualifies the threshold.
    """
    answerable = [cut for has_relevant, cut in rows if has_relevant]
    if any(cut.forbidden for _has_relevant, cut in rows):
        return -1.0, -1.0, -1.0, -threshold
    if not answerable:
        return 0.0, 0.0, 0.0, -threshold
    mrr = sum(1 / cut.best_rank for cut in answerable if cut.best_rank) / len(answerable)
    recall_at_3 = sum(1 for cut in answerable if cut.best_rank and cut.best_rank <= 3) / len(answerable)
    precision = sum(cut.precision for cut in answerable) / len(answerable)
    return mrr, recall_at_3, precision, -threshold


def calibrate_fixture_file(path, conn, mode: str, *, embedder=None, require_usable: bool = False) -> CalibrationRecord:
    """Choose the best observed score threshold from a held-out fixture.

    ``lexical_plus_embedder`` evaluates in that mode, sweeping the fused and
    rescored scores, and needs the configured embedder; its version also
    covers the embedder fingerprint, because a threshold belongs to a model.
    ``require_usable`` (``recall-calibrate --apply``) refuses a degenerate
    result: no query with relevant documents, forbidden results at every
    threshold, or no relevant document retrieved at any threshold.
    """
    fixture_path = Path(path)
    if _is_public_fixture(fixture_path):
        raise ValueError("public evaluation fixtures cannot be used for calibration")
    if mode == EMBEDDER_MODE and embedder is None:
        raise ValueError(f"calibrating {EMBEDDER_MODE} needs the configured embedder")
    data = _fixture(fixture_path)
    if require_usable and not any(_relevant(item) for item in data["queries"]):
        raise ValueError("refusing --apply: the fixture has no query with relevant documents")
    settings = _default_settings()
    _ensure_documents(conn, data["documents"])
    embedder_mode = mode == EMBEDDER_MODE
    if embedder_mode:
        report = _embed_fixture_documents(conn, embedder, settings)
        if report is None or report.error:
            raise ValueError(f"could not embed the fixture documents: {report.error if report else 'embedder failed'}")
        run_settings = _embedder_settings(settings)
    else:
        embedder = None
        run_settings = _lexical_settings(settings)
    # Rank every query once; each threshold then only looks up its selection.
    per_query = []
    scores: set[float] = set() if embedder_mode else {0.0}
    for item in data["queries"]:
        query = _query(item, settings)
        outcome = retrieve(conn, query, run_settings, {}, embedder=embedder, require_calibration=False)
        if embedder_mode and outcome.fallback_reason:
            raise ValueError(f"embedder mode fell back during calibration: {outcome.fallback_reason}")
        cuts, rows = _query_cuts(item, query, outcome, run_settings, mode)
        per_query.append((bool(_relevant(item)), cuts, rows))
        scores.update(cuts)
    best: tuple[tuple[float, float, float, float], float] | None = None
    swept = sorted(scores)
    for index, threshold in enumerate(swept):
        rows = [(has_relevant, cut_rows[bisect.bisect_left(cuts, threshold)]) for has_relevant, cuts, cut_rows in per_query]
        key = calibration_key(rows, threshold)
        if best is None or key > best[0]:
            # Every value in (previous swept score, threshold] selects the same
            # fixture results; report the middle of that interval, so a score
            # that moves slightly on the live index does not fall off the edge.
            lower = swept[index - 1] if index else 0.0
            best = (key, (lower + threshold) / 2 if threshold > 0 else threshold)
    if require_usable:
        if best is None or best[0][0] < 0:
            raise ValueError("refusing --apply: every threshold returns a forbidden result")
        if best[0][0] <= 0:
            raise ValueError("refusing --apply: no relevant document was retrieved at any threshold")
        if embedder_mode and best[1] <= 0.0:
            raise ValueError(f"refusing --apply: a {EMBEDDER_MODE} threshold of 0 would select every dense candidate")
    canonical_hash = hashlib.sha256(_canonical(data)).hexdigest()
    if embedder is not None:
        version = embedder_calibration_version(canonical_hash, mode, adapter_fingerprint(embedder))
    else:
        version = hashlib.sha256(f"{canonical_hash}:{mode}:calibration-v1".encode("utf-8")).hexdigest()
    return CalibrationRecord(mode, version, best[1] if best else 0.0, canonical_hash, dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"))
