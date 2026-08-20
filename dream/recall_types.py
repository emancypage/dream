"""Immutable contracts for automatic memory recall."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterSettings:
    """Configuration of one optional recall adapter (embedder or reranker)."""

    enabled: bool
    type: str
    remote_data_egress: bool


@dataclass(frozen=True)
class RecallSettings:
    """Core recall configuration plus its optional adapter settings."""

    enabled: bool
    install_hooks: bool
    allow_raw_transcript_prompt: bool
    first_prompt_only: bool
    session_start_budget_codepoints: int
    prompt_budget_codepoints: int
    session_start_additional_context_limit: int
    prompt_additional_context_limit: int
    diagnostic_path: str
    calibration_path: str
    embedder: AdapterSettings
    reranker: AdapterSettings


@dataclass(frozen=True)
class RecallDocument:
    """A canonical, searchable recall document (identity plus content)."""

    id: str
    content_sha256: str
    source_kind: str
    trust_level: str
    project_slug: str | None
    source_path: str
    source_updated_at: str
    indexed_at: str
    source_version: str
    text: str


@dataclass(frozen=True)
class RecallQuery:
    """A retrieval request produced by a hook event or manual invocation."""

    query_text: str
    session_id: str | None
    hook_event: str
    cwd: str | None
    repository_roots: tuple[str, ...]
    requested_codepoint_budget: int
    excluded_source_ids: frozenset[str]
    allow_raw_transcript: bool


@dataclass(frozen=True)
class RecallCandidate:
    """A retrieved document with its scrubbed excerpt and ranking evidence."""

    document: RecallDocument
    scrubbed_excerpt: str
    component_ranks: Mapping[str, int]
    score: float
    best_component_rank: int


@dataclass(frozen=True)
class RecallDiagnostics:
    """Bounded, structured diagnostics for one recall run."""

    candidate_count: int
    selected_count: int
    fallback_reason: str | None
    selected_codepoints: int
    elapsed_ms: int
    calibration_version: str | None


@dataclass(frozen=True)
class RecallResult:
    """The outcome of one recall run: selection, rendered context, and diagnostics."""

    selected: tuple[RecallCandidate, ...]
    rendered_context: str
    mode: str
    calibration_version: str | None
    diagnostics: RecallDiagnostics


@dataclass(frozen=True)
class CalibrationRecord:
    """A calibrated score threshold for one retrieval mode."""

    mode: str
    calibration_version: str
    threshold: float
    fixture_sha256: str
    created_at: str
