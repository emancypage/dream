"""Fail-open context execution, diagnostics, and preflight."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from recall_events import POLICY_VERSION
from recall_adapters import AdapterError, EmbedderUnavailable, adapter_fingerprint, apply_optional_adapters, build_embedder, embed_query, embedder_fallback_reason
from recall_embeddings import dense_candidates, embedding_text, has_vectors
from recall_render import _render_blocks
from recall_query import _load_documents, rank_lexical
from recall_scrub import scrub_text
from recall_select import calibration_for, embedder_calibration_version, select_recall_candidates
from recall_types import RecallCandidate, RecallDiagnostics, RecallQuery, RecallResult
from recall_usage import usage_map
from storage import open_db_readonly
from retention_policy import load_retention_policy


def parse_hook_payload(event_name: str, payload: dict) -> RecallQuery:
    if not isinstance(payload, dict):
        raise ValueError("hook payload must be an object")
    session_id = payload.get("session_id")
    cwd = payload.get("cwd")
    roots = payload.get("repository_roots") or ()
    if not session_id or not cwd:
        raise ValueError("session_id and cwd are required")
    if event_name == "UserPromptSubmit":
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("prompt is required")
        return RecallQuery(prompt, str(session_id), "prompt", str(cwd), tuple(str(root) for root in roots), 4000, frozenset(), False)
    source = payload.get("source") or payload.get("event") or "startup"
    return RecallQuery(
        " ".join([str(cwd), *(str(root) for root in roots)]),
        str(session_id), f"session-start:{source}", str(cwd), tuple(str(root) for root in roots), 6000, frozenset(), False,
    )


def _load_calibrations(conn):
    return {
        mode: {"threshold": threshold, "calibration_version": version, "fixture_sha256": fixture_sha256}
        for mode, version, threshold, fixture_sha256 in conn.execute(
            "SELECT mode, calibration_version, threshold, fixture_sha256 FROM recall_calibrations").fetchall()
    }


def _calibration_is_current(row, fingerprint: str) -> bool:
    if not isinstance(row, dict):
        return False
    expected = embedder_calibration_version(str(row.get("fixture_sha256") or ""), EMBEDDER_MODE, fingerprint)
    return row.get("calibration_version") == expected


EMBEDDER_MODE = "lexical_plus_embedder"
PREFLIGHT_PROBE = "dream preflight probe"


@dataclass(frozen=True)
class Retrieval:
    """Ranked and selected candidates of one recall pipeline run."""

    candidates: tuple[RecallCandidate, ...]
    selected: tuple[RecallCandidate, ...]
    mode: str
    fallback_reason: str | None
    calibration_version: str | None


def with_adapters(settings, *, embedder: bool | None = None, reranker: bool | None = None):
    """Settings with the embedder and/or the reranker switched on or off (None keeps it)."""
    if embedder is not None:
        settings = replace(settings, embedder=replace(settings.embedder, enabled=embedder))
    if reranker is not None:
        settings = replace(settings, reranker=replace(settings.reranker, enabled=reranker))
    return settings


def _dense_list(conn, query: RecallQuery, settings, embedder, calibrations, *, require_calibration: bool):
    """Return ``(dense, query_vector, fallback_reason)`` for an enabled embedder.

    The query text is embedded at most once, and only after every local
    check passed. Unless the embedder mode has a calibration row computed for
    the current embedder fingerprint, the dense list is dropped before any
    network call: fused scores are on another scale than the lexical
    threshold, a threshold from another model does not fit this one, and
    without a threshold every prompt would inject up to its budget. A store
    without vectors for the fingerprint is not queried either. Every failure
    leaves ``dense`` None, so ranking stays lexical.
    """
    if query.hook_event.startswith("session-start"):
        # The cwd and roots carry no topic, and session-start selection has no
        # threshold: a dense list there would inject unrelated documents.
        return None, None, None
    if embedder is None:
        return None, None, "embedder-unavailable"
    try:
        fingerprint = adapter_fingerprint(embedder)
    except Exception as exc:
        return None, None, embedder_fallback_reason(exc)
    if require_calibration:
        if EMBEDDER_MODE not in calibrations:
            return None, None, "embedder-uncalibrated"
        if not _calibration_is_current(calibrations[EMBEDDER_MODE], fingerprint):
            return None, None, "embedder-calibration-stale"
    if not query.query_text.strip():
        return None, None, None
    kinds = ("approved_memory", "distilled_summary") + (("raw_transcript",) if query.allow_raw_transcript else ())
    try:
        if not has_vectors(conn, fingerprint, kinds):
            return None, None, "embedder-no-vectors"
        # Scrubbed like the stored documents: not even a loopback server sees a credential.
        query_vector = embed_query(embedder, embedding_text(query.query_text))
        dense = dense_candidates(conn, query_vector, fingerprint, kinds, limit=settings.embedder.dense_candidates)
    except Exception as exc:
        return None, None, embedder_fallback_reason(exc)
    if not dense:
        return None, None, "embedder-no-vectors"
    return dense, query_vector, None


def retrieve(conn, query: RecallQuery, settings, calibrations, *, embedder=None, reranker=None,
             allow_cache_writes: bool = False, keep=None, require_calibration: bool = True) -> Retrieval:
    """Rank, fuse the optional dense list, rescore and select with the active mode's calibration.

    ``keep`` filters ranked candidates (retention policy, constitution) before
    adapters and selection. The recall hooks and the evaluator share this
    pipeline, so a fixture measures exactly what a session receives. Only the
    evaluator and calibration, which compute the embedder mode's threshold,
    pass ``require_calibration=False``.
    """
    dense = query_vector = fallback = None
    adapter_settings = settings
    if settings.embedder.enabled:
        dense, query_vector, fallback = _dense_list(conn, query, settings, embedder, calibrations,
                                                    require_calibration=require_calibration)
        if query_vector is None:
            # Nothing dense contributes: no second embedder call, lexical scores stay.
            adapter_settings = with_adapters(settings, embedder=False)
    # Usage only reorders exact ties, and only once switched on; an unmigrated
    # or read-only database without recall_usage simply yields no usage.
    usage = usage_map(conn) if getattr(settings, "usage_tie_break", False) else None
    candidates = rank_lexical(conn, query, dense=dense, usage=usage)
    if keep is not None:
        candidates = [candidate for candidate in candidates if keep(candidate)]
    candidates, mode, adapter_fallback = apply_optional_adapters(
        conn, query, candidates, adapter_settings, embedder=embedder, reranker=reranker,
        allow_cache_writes=allow_cache_writes, query_vector=query_vector, usage=usage,
    )
    if mode == "lexical" and any("dense" in candidate.component_ranks for candidate in candidates):
        # Fused dense ranks change the score scale even when nothing was rescored.
        mode = EMBEDDER_MODE
    threshold, version = calibration_for(calibrations, mode, query)
    selected = select_recall_candidates(candidates, query, settings, calibrations, mode=mode, threshold=threshold)
    return Retrieval(tuple(candidates), selected, mode, fallback or adapter_fallback, version)


def constitution_guard(memory_root, constitution_path):
    """``character.safety.is_constitution_source`` for many documents, resolving the constitution once.

    Returns ``is_constitution(source_path) -> bool``. Each source path is
    still resolved and, when that differs from the constitution, compared by
    file identity, so a symlink or hard link to the constitution is caught
    exactly as ``is_constitution_source`` catches it. The constitution's path
    and identity are read on the first call.
    """
    if constitution_path is None:
        return lambda source_path: False
    target = Path(constitution_path).expanduser()
    resolved: list = []

    def is_constitution(source_path) -> bool:
        source = Path(source_path).expanduser()
        if not source.is_absolute() and memory_root is not None:
            source = Path(memory_root) / source
        source_resolved = source.resolve()
        if not resolved:
            try:
                target_stat = target.stat()
            except OSError:
                target_stat = None
            resolved.append((target.resolve(), target_stat))
        target_resolved, target_stat = resolved[0]
        if source_resolved == target_resolved:
            return True
        if target_stat is None:
            return False
        try:
            return os.path.samestat(source.stat(), target_stat)
        except OSError:
            return False

    return is_constitution


def document_allowed(document, *, policy, memory_root, constitution_path, is_constitution=None) -> bool:
    """Whether retention policy and the constitution guard let recall serve ``document``.

    ``is_constitution`` is a ``constitution_guard`` shared by one run's
    documents; without it, ``is_constitution_source`` checks this one.
    """
    if is_constitution is None:
        from character.safety import is_constitution_source

        def is_constitution(source_path) -> bool:
            return is_constitution_source(source_path, memory_root=memory_root, constitution_path=constitution_path)
    return not (
        is_constitution(document.source_path)
        or policy.excludes(source_path=document.source_path, project_slug=document.project_slug, text=document.text)
        or (document.source_kind == "approved_memory" and memory_root is not None
            and policy.excludes_memory_path(memory_root, document.source_path))
    )


def run_context(conn, query: RecallQuery, settings, explain: bool = False, *, embedder=None, reranker=None, allow_cache_writes: bool = False, memory_root=None, constitution_path,
                kinds: frozenset[str] | None = None, limit: int | None = None) -> RecallResult:
    """Retrieve, select and render one recall run.

    ``kinds`` is a hard source-kind filter applied before selection, and
    ``limit`` keeps the first selected candidates before rendering. Both
    default to None (every kind, no limit), which the hooks rely on.
    """
    return _run_context(conn, query, settings, explain, embedder=embedder, reranker=reranker,
                        allow_cache_writes=allow_cache_writes, memory_root=memory_root,
                        constitution_path=constitution_path, kinds=kinds, limit=limit)[0]


def _run_context(conn, query: RecallQuery, settings, explain: bool = False, *, embedder=None, reranker=None,
                 allow_cache_writes: bool = False, memory_root=None, constitution_path, kinds=None, limit=None,
                 ) -> tuple[RecallResult, tuple[RecallCandidate, ...]]:
    """``run_context`` plus the selected candidates whose block was rendered."""
    started = time.perf_counter()
    policy = load_retention_policy(memory_root)
    is_constitution = constitution_guard(memory_root, constitution_path)

    def keep(candidate) -> bool:
        if kinds is not None and candidate.document.source_kind not in kinds:
            return False
        return document_allowed(candidate.document, policy=policy, memory_root=memory_root,
                                constitution_path=constitution_path, is_constitution=is_constitution)

    outcome = retrieve(
        conn, query, settings, _load_calibrations(conn), embedder=embedder, reranker=reranker,
        allow_cache_writes=allow_cache_writes, keep=keep,
    )
    selected = outcome.selected if limit is None else outcome.selected[:limit]
    rendered, kept = _render_blocks(selected, query.requested_codepoint_budget, query.query_text)
    elapsed = int((time.perf_counter() - started) * 1000)
    diagnostics = RecallDiagnostics(len(outcome.candidates), len(selected), outcome.fallback_reason,
                                    len(rendered), elapsed, outcome.calibration_version)
    return RecallResult(selected, rendered, outcome.mode, outcome.calibration_version, diagnostics), tuple(kept)


AGENT_QUERY_EVENT = "agent-query"
AGENT_QUERY_MAX_LIMIT = 10
RECALL_KINDS = ("approved_memory", "distilled_summary", "raw_transcript")
# Agent queries read approved memory and distilled summaries unless raw transcripts are asked for explicitly.
AGENT_DEFAULT_KINDS = frozenset({"approved_memory", "distilled_summary"})
# Every agent query reads these. recall_embeddings is optional: without it ranking stays lexical.
AGENT_RECALL_TABLES = ("recall_documents", "recall_documents_fts", "recall_calibrations")


class RecallUnavailable(Exception):
    """The store cannot answer an agent query: recall is disabled or the schema is missing."""


def missing_recall_tables(conn) -> tuple[str, ...]:
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return tuple(name for name in AGENT_RECALL_TABLES if name not in names)


def validate_agent_query(query_text, kinds, settings) -> tuple[str, frozenset[str]]:
    """Check one agent query's user input: ``(query_text, kinds)`` or ValueError.

    ``kinds`` None means approved memory and distilled summaries, even when
    raw transcript recall is enabled. An explicit ``raw_transcript`` is
    accepted only when ``recall.allow_raw_transcript_prompt`` is true.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("expected a non-empty query")
    if kinds is None:
        return query_text, AGENT_DEFAULT_KINDS
    if not isinstance(kinds, (list, tuple, set, frozenset)):
        raise ValueError("kinds must be a list of kind names")
    names = tuple(dict.fromkeys(kind.strip() if isinstance(kind, str) else repr(kind)[:40] for kind in kinds))
    names = tuple(name for name in names if name)
    if not names:
        raise ValueError("expected at least one kind")
    unknown = [name[:40] for name in names if name not in RECALL_KINDS]
    if unknown:
        raise ValueError(f"unknown kind {', '.join(unknown[:3])}; expected {', '.join(RECALL_KINDS)}")
    if "raw_transcript" in names and not settings.allow_raw_transcript_prompt:
        raise ValueError("raw_transcript recall is disabled; it needs recall.allow_raw_transcript_prompt = true")
    return query_text, frozenset(names)


def recall_for_agent(conn, settings, *, query_text, cwd, memory_root, constitution_path, project_slug=None,
                     kinds=None, limit=5, budget_codepoints=4000, embedder=None) -> RecallResult:
    """Answer an agent's memory query: the entry point of ``dream recall`` and the MCP server.

    The query takes the prompt path with the ``agent-query`` event, so the
    active mode's calibrated threshold and the embedder calibration gate
    apply, and runs with ``allow_cache_writes=False``. ``conn`` should be a
    read-only connection; nothing here writes markers, cache rows or files.
    ``project_slug`` joins the path evidence as a soft boost: every project
    stays eligible. ``kinds`` is a hard filter that defaults to approved
    memory and distilled summaries; raw transcripts need an explicit
    ``raw_transcript`` and the opt-in. ``limit`` is clamped to 1-10.
    ``selected`` holds the candidates whose block is in the rendered
    context. Invalid input raises ValueError; a disabled or unmigrated store
    raises RecallUnavailable.
    """
    query_text, allowed = validate_agent_query(query_text, kinds, settings)
    if project_slug is not None and not isinstance(project_slug, str):
        raise ValueError("project must be a string")
    project = (project_slug or "").strip()
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    limit = max(1, min(AGENT_QUERY_MAX_LIMIT, limit))
    if not settings.enabled:
        raise RecallUnavailable("recall disabled")
    missing = missing_recall_tables(conn)
    if missing:
        raise RecallUnavailable("recall tables missing: " + ", ".join(missing))
    allow_raw = settings.allow_raw_transcript_prompt and "raw_transcript" in allowed
    query = RecallQuery(query_text, None, AGENT_QUERY_EVENT, str(cwd) if cwd else None, (project,) if project else (),
                        int(budget_codepoints), frozenset(), allow_raw)
    result, shown = _run_context(conn, query, settings, embedder=embedder, allow_cache_writes=False,
                                 memory_root=memory_root, constitution_path=constitution_path,
                                 kinds=allowed, limit=limit)
    return replace(result, selected=shown, diagnostics=replace(result.diagnostics, selected_count=len(shown)))


def _agent_store_ready(conn, settings) -> None:
    if not settings.enabled:
        raise RecallUnavailable("recall disabled")
    missing = missing_recall_tables(conn)
    if missing:
        raise RecallUnavailable("recall tables missing: " + ", ".join(missing))


def _as_candidate(document) -> RecallCandidate:
    return RecallCandidate(document, scrub_text(document.text, Path.home()), {}, 0.0, 0)


def document_for_agent(conn, settings, document_id: str, *, memory_root, constitution_path) -> RecallCandidate:
    """One current recall document by ID, under the checks every agent query applies.

    Retention policy and the constitution guard filter it exactly as in
    ``run_context``, and a ``raw_transcript`` needs
    ``recall.allow_raw_transcript_prompt``. The candidate's excerpt is
    scrubbed; callers still render it through ``render_context``. An unknown
    or excluded ID raises LookupError with the same message, so the error
    never confirms that an excluded document exists. A disabled or unmigrated
    store raises RecallUnavailable.
    """
    _agent_store_ready(conn, settings)
    documents = _load_documents(conn, [document_id])
    policy = load_retention_policy(memory_root)
    if (not documents or documents[0].source_kind not in RECALL_KINDS
            or (documents[0].source_kind == "raw_transcript" and not settings.allow_raw_transcript_prompt)
            or not document_allowed(documents[0], policy=policy, memory_root=memory_root,
                                    constitution_path=constitution_path)):
        raise LookupError("document not found or not available for recall")
    return _as_candidate(documents[0])


MEMORY_INDEX_PATH = "MEMORY.md"


def memory_index_for_agent(conn, settings, *, memory_root, constitution_path) -> tuple[RecallCandidate | None, int]:
    """The indexed ``MEMORY.md`` document and the number of other approved memory documents.

    Reads the database only, never the memory root's files: the index is the
    synchronized recall document, and the count covers indexed
    ``approved_memory`` documents that recall may serve, not files on disk.
    The candidate is None when the store holds no servable index document.
    """
    _agent_store_ready(conn, settings)
    ids = [row[0] for row in conn.execute("SELECT id FROM recall_documents WHERE source_kind = 'approved_memory'")]
    policy = load_retention_policy(memory_root)
    servable = [document for document in _load_documents(conn, ids)
                if document_allowed(document, policy=policy, memory_root=memory_root, constitution_path=constitution_path)]
    index = next((document for document in servable if document.source_path == MEMORY_INDEX_PATH), None)
    others = sum(1 for document in servable if document.source_path != MEMORY_INDEX_PATH)
    return (_as_candidate(index) if index is not None else None), others


def agent_results(result: RecallResult) -> list[dict]:
    """The structured result rows of an agent query: ``id``, ``kind``, ``trust``, ``project``, ``score``."""
    return [
        {"id": candidate.document.id, "kind": candidate.document.source_kind, "trust": candidate.document.trust_level,
         "project": candidate.document.project_slug, "score": candidate.score}
        for candidate in result.selected
    ]


def embedder_config_errors(config) -> tuple[str, ...]:
    return tuple(error for error in config.optional_config_errors if "recall.embedder" in error)


def configured_embedder(config):
    """Build the configured embedder without contacting it: ``(embedder, problem)``.

    ``(None, None)`` means the embedder is off. A problem means it was
    requested but cannot be used (invalid configuration, or a non-loopback
    URL without ``remote_data_egress``); callers then run lexical recall and
    report the problem.
    """
    errors = embedder_config_errors(config)
    raw = (config.data.get("recall") or {}).get("embedder")
    requested = isinstance(raw, dict) and raw.get("enabled") is True
    if errors:
        return None, ("; ".join(errors) if requested else None)
    try:
        return build_embedder(config.recall_settings().embedder), None
    except AdapterError as exc:
        return None, str(exc)


def embedder_preflight(config) -> tuple[bool, str]:
    """Report the embedder: disabled, misconfigured, unreachable or reachable.

    Reachability sends one fixed probe string, never a document.
    """
    errors = embedder_config_errors(config)
    if errors:
        return False, "misconfigured: " + "; ".join(errors)
    settings = config.recall_settings().embedder
    if not settings.enabled:
        return True, "disabled"
    try:
        embedder = build_embedder(settings)
    except AdapterError as exc:
        return False, f"misconfigured: {exc}"
    if embedder is None:
        return True, f"adapter type {settings.type}"
    try:
        vectors = embedder.embed([PREFLIGHT_PROBE])
    except EmbedderUnavailable as exc:
        return False, f"unreachable: {exc}"
    except AdapterError as exc:
        return False, f"misconfigured: {exc}"
    except Exception as exc:
        return False, f"unreachable: {type(exc).__name__}"
    model = embedder.server_model or f"{settings.model} (not reported by the server)"
    scope = "remote" if embedder.remote_data_egress else "loopback"
    return True, f"reachable: server model {model}, dimension {len(vectors[0])}, {scope}"


def hook_success_json(hook_event_name: str, context: str) -> str:
    return json.dumps({
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "additionalContext": context,
        },
    }, ensure_ascii=False, separators=(",", ":"))


def recent_diagnostic(settings, fallback_reason: str, window_seconds: float = 600.0) -> bool:
    """Whether the last diagnostic line, written within the window, has this fallback reason.

    Lets a hook report a down embedder once per window instead of once per prompt.
    """
    path = Path(settings.diagnostic_path).expanduser()
    try:
        status = path.stat()
        if time.time() - status.st_mtime > window_seconds:
            return False
        with path.open("rb") as handle:
            handle.seek(max(0, status.st_size - 9000))
            lines = handle.read().splitlines()
        return bool(lines) and json.loads(lines[-1]).get("fallback_reason") == fallback_reason
    except (OSError, ValueError, AttributeError):
        return False


def append_diagnostic(settings, diagnostic: dict) -> None:
    path = Path(settings.diagnostic_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"))[:8000]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def codex_memories_check(codex_home: Path | None = None) -> tuple[bool, str]:
    home = Path.home() / ".codex" if codex_home is None else Path(codex_home)
    config_path = home / "config.toml"
    config = {}
    try:
        if config_path.exists():
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
    except (UnicodeError, OSError, tomllib.TOMLDecodeError) as exc:
        return False, f"Codex Memories configuration could not be read: {exc}"

    features = config.get("features", {})
    memories = config.get("memories", {})
    if not isinstance(features, dict):
        return False, "Codex Memories configuration could not be read: [features] must be a table"
    if not isinstance(memories, dict):
        return False, "Codex Memories configuration could not be read: [memories] must be a table"
    feature_enabled = features.get("memories") is True
    use_memories = memories.get("use_memories")
    generate_memories = memories.get("generate_memories")
    all_controls_disabled = use_memories is False and generate_memories is False
    if feature_enabled and not all_controls_disabled:
        controls = ["features.memories"]
        if use_memories is True:
            controls.append("memories.use_memories")
        if generate_memories is True:
            controls.append("memories.generate_memories")
        return False, (
            "Codex Memories enabled by " + ", ".join(controls) +
            "; disable native Codex Memories before relying on Dream"
        )

    memories_path = home / "memories"
    if memories_path.is_dir():
        payload_files = [
            path for path in memories_path.rglob("*")
            if path.is_file() and not ({*path.relative_to(memories_path).parts} & {".agents", ".codex"})
        ]
        if not payload_files:
            return True, "Codex Memories is disabled or has an empty/stale scaffold"
    return True, "native Codex Memories disabled"


def mcp_install_status() -> tuple[bool, str]:
    """Where the Dream MCP server entry is installed, from ``mcp_install.status``; read-only.

    Checks the Codex and OpenCode configuration files that ``dream mcp
    install`` edits and writes nothing. Claude Code keeps its own registry,
    which only ``claude mcp get dream`` shows, so it is named, not checked.
    Always ``ok``: the server is optional.
    """
    import mcp_install

    installed: list[str] = []
    notes: list[str] = []
    for client in mcp_install.FILE_CLIENTS:
        try:
            state = mcp_install.status(client)
        except mcp_install.McpInstallError as exc:
            notes.append(f"{client}: {' '.join(str(exc).split())[:160]}")
            continue
        if state.startswith("installed"):
            installed.append(client if state == "installed" else f"{client} ({state.split(', ', 1)[1]})")
        elif state != "not installed":
            notes.append(f"{client}: {state}")
    detail = "installed for " + ", ".join(installed) if installed else "not installed for codex or opencode"
    return True, "; ".join([detail, *notes, "for Claude Code run `claude mcp get dream`"])


def recall_preflight(config, db_path) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    path = Path(db_path).expanduser()
    checks.append(("recall.schema", path.exists(), "database exists" if path.exists() else "database does not exist"))
    if path.exists():
        try:
            conn = open_db_readonly(path)
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')")}
            ready = {"recall_documents", "recall_documents_fts", "recall_events"}.issubset(names)
            checks.append(("recall.index", ready, "recall schema present" if ready else "recall schema incomplete"))
            conn.close()
        except sqlite3.Error as exc:
            checks.append(("recall.index", False, str(exc)))
    else:
        checks.append(("recall.index", False, "database does not exist"))
    checks.append(("recall.embedder", *embedder_preflight(config)))
    reranker = config.recall_settings().reranker
    checks.append(("recall.reranker", not reranker.enabled or reranker.type == "none", "disabled" if not reranker.enabled else f"adapter type {reranker.type}"))
    checks.append(("recall.mcp", *mcp_install_status()))
    executable = shutil.which("codex")
    version_ok = False
    detail = "codex executable not found"
    if executable:
        try:
            output = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=1, check=False).stdout.strip()
            version_ok = bool(output)
            detail = output or "codex version unavailable"
        except (OSError, subprocess.SubprocessError) as exc:
            detail = str(exc)
    checks.append(("codex.version", version_ok, detail))
    memories_ok, memories_detail = codex_memories_check()
    checks.append(("codex.memories-double-injection", memories_ok, memories_detail))
    return checks


def compose_hook_context(recall_text, character_text, codepoint_limit, additional_context_limit, *, reminder_text=''):
    """Keep complete character; conservatively bound tokens by UTF-8 bytes."""
    if not character_text and not reminder_text:
        return recall_text, None
    limit = min(codepoint_limit, additional_context_limit)
    reason = None
    if len(reminder_text.encode('utf-8')) > limit:
        return recall_text, 'review-reminder-over-budget'
    prefix = reminder_text + ('\n\n' if reminder_text and character_text else '')
    if len((prefix + character_text).encode('utf-8')) > limit:
        character_text = ''
        reason = 'character-over-budget'
    trusted = reminder_text + ('\n\n' if reminder_text and character_text else '') + character_text
    if not trusted:
        return recall_text, reason
    separator = '\n\n# Untrusted recall\n'
    remaining = limit - len((trusted + separator).encode('utf-8'))
    if remaining <= 0:
        return trusted, reason
    recall_text = recall_text.encode('utf-8')[:remaining].decode('utf-8', errors='ignore')
    return trusted + (separator + recall_text if recall_text else ''), reason
