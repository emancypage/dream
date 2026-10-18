"""Deterministic policy selection on top of lexical or optional rankings."""

from __future__ import annotations

from collections.abc import Mapping

from recall_types import RecallCandidate, RecallQuery, RecallSettings


def _threshold(calibrations, mode: str) -> float | None:
    if not calibrations:
        return None
    value = calibrations.get(mode) if isinstance(calibrations, Mapping) else None
    if value is None:
        return None
    if hasattr(value, "threshold"):
        return float(value.threshold)
    if isinstance(value, Mapping) and value.get("threshold") is not None:
        return float(value["threshold"])
    return None


def select_recall_candidates(candidates, query, settings, calibrations) -> tuple[RecallCandidate, ...]:
    """Apply threshold, provenance, deduplication, and source caps in rank order."""
    mode = "lexical"
    threshold = None if query.hook_event.startswith("session-start") else _threshold(calibrations, mode)
    filtered = []
    seen_hashes: set[str] = set()
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        doc = candidate.document
        if doc.id in query.excluded_source_ids:
            continue
        if doc.source_kind == "raw_transcript" and not query.allow_raw_transcript:
            continue
        if threshold is not None and candidate.score < threshold:
            continue
        if doc.content_sha256 in seen_hashes:
            continue
        source = doc.source_path
        if source_counts.get(source, 0) >= 2:
            continue
        seen_hashes.add(doc.content_sha256)
        source_counts[source] = source_counts.get(source, 0) + 1
        filtered.append(candidate)

    return tuple(filtered)
