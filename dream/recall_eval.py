"""Fixture-based evaluation and held-out threshold calibration."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path

from recall_query import rank_lexical
from recall_render import render_context
from recall_select import select_recall_candidates
from recall_types import CalibrationRecord, RecallQuery, RecallSettings


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

    @property
    def metrics(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return asdict(self)


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


def _evaluate(data: dict, conn, settings: RecallSettings, threshold: float | None = None) -> EvaluationReport:
    _ensure_documents(conn, data["documents"])
    rows: list[dict] = []
    latencies: list[float] = []
    all_forbidden = 0
    duplicate_count = 0
    total_codepoints = 0
    empty_correct: list[float] = []
    mode = "lexical"
    calibration = {mode: {"threshold": threshold}} if threshold is not None else {}
    for item in data["queries"]:
        query = _query(item, settings)
        started = time.perf_counter()
        candidates = rank_lexical(conn, query)
        selected = select_recall_candidates(candidates, query, settings, calibration)
        rendered = render_context(selected, query.requested_codepoint_budget)
        latencies.append((time.perf_counter() - started) * 1000)
        selected_ids = [candidate.document.id for candidate in selected]
        relevant = set(item.get("relevant", ()))
        forbidden = set(item.get("forbidden", ()))
        all_forbidden += len(forbidden.intersection(selected_ids))
        duplicate_count += len(selected_ids) - len({candidate.document.content_sha256 for candidate in selected})
        total_codepoints += len(rendered)
        if relevant:
            ranks = [index + 1 for index, doc_id in enumerate(selected_ids) if doc_id in relevant]
            rows.append({"ranks": ranks, "relevant": relevant})
        else:
            empty_correct.append(1.0 if not selected_ids else 0.0)
    query_count = len(data["queries"])
    def at(k: int) -> float:
        return sum(1 for row in rows if any(rank <= k for rank in row["ranks"])) / len(rows) if rows else 0.0
    mrr = sum(1 / min(row["ranks"]) for row in rows if row["ranks"]) / len(rows) if rows else 0.0
    injected = [row for row in data["queries"] if row.get("injected")]
    injected_hits = sum(1 for row in injected if set(row.get("injected", ())).intersection({candidate.document.id for candidate in rank_lexical(conn, _query(row, settings))}))
    injected_precision = injected_hits / len(injected) if injected else 1.0
    return EvaluationReport(
        at(1), at(3), at(5), mrr, injected_precision, all_forbidden,
        statistics.mean(empty_correct) if empty_correct else 1.0,
        duplicate_count, total_codepoints,
        statistics.median(latencies) if latencies else 0.0,
        sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        None, query_count,
    )


def evaluate_fixture_file(path, conn, settings=None) -> EvaluationReport:
    """Evaluate a fixture against the supplied database."""
    return _evaluate(_fixture(Path(path)), conn, settings or _default_settings())


def _score(record: EvaluationReport) -> tuple[float, float]:
    return record.mrr, record.recall_at_3


def calibrate_fixture_file(path, conn, mode: str) -> CalibrationRecord:
    """Choose the best observed score threshold from a held-out fixture."""
    fixture_path = Path(path)
    if fixture_path.name.lower() == "public.json" or "public" in {part.lower() for part in fixture_path.parts}:
        raise ValueError("public evaluation fixtures cannot be used for calibration")
    data = _fixture(fixture_path)
    settings = _default_settings()
    _ensure_documents(conn, data["documents"])
    scores = {0.0}
    for item in data["queries"]:
        scores.update(candidate.score for candidate in rank_lexical(conn, _query(item, settings)))
    best: tuple[tuple[float, float, float], float, EvaluationReport] | None = None
    for threshold in sorted(scores):
        report = _evaluate(data, conn, settings, threshold)
        key = (report.mrr if report.forbidden_count == 0 else -1.0, report.recall_at_3 if report.forbidden_count == 0 else -1.0, -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, report)
    canonical_hash = hashlib.sha256(_canonical(data)).hexdigest()
    version = hashlib.sha256(f"{canonical_hash}:{mode}:calibration-v1".encode("utf-8")).hexdigest()
    return CalibrationRecord(mode, version, best[1] if best else 0.0, canonical_hash, dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"))
