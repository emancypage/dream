"""Fail-open context execution, diagnostics, and preflight."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
import tomllib
from dataclasses import replace
from pathlib import Path

from recall_events import POLICY_VERSION
from recall_adapters import apply_optional_adapters
from recall_render import render_context
from recall_query import rank_lexical
from recall_select import select_recall_candidates
from recall_types import RecallDiagnostics, RecallQuery, RecallResult
from storage import open_db_readonly


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
        mode: {"threshold": threshold, "calibration_version": version}
        for mode, version, threshold in conn.execute("SELECT mode, calibration_version, threshold FROM recall_calibrations").fetchall()
    }


def run_context(conn, query: RecallQuery, settings, explain: bool = False, *, embedder=None, reranker=None, allow_cache_writes: bool = True) -> RecallResult:
    started = time.perf_counter()
    candidates = rank_lexical(conn, query)
    candidates, mode, fallback = apply_optional_adapters(
        conn, query, candidates, settings, embedder=embedder, reranker=reranker, allow_cache_writes=allow_cache_writes,
    )
    selected = select_recall_candidates(candidates, query, settings, _load_calibrations(conn))
    rendered = render_context(selected, query.requested_codepoint_budget)
    elapsed = int((time.perf_counter() - started) * 1000)
    diagnostics = RecallDiagnostics(len(candidates), len(selected), fallback, len(rendered), elapsed, None)
    return RecallResult(tuple(selected), rendered, mode, None, diagnostics)


def hook_success_json(context: str) -> str:
    return json.dumps({"continue": True, "hookSpecificOutput": {"additionalContext": context}}, ensure_ascii=False, separators=(",", ":"))


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
    if config_path.exists():
        try:
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            return False, f"Codex Memories configuration could not be read: {exc}"

    features = config.get("features", {})
    memories = config.get("memories", {})
    feature_enabled = features.get("memories") is True
    use_memories = memories.get("use_memories")
    generate_memories = memories.get("generate_memories")
    explicitly_disabled = use_memories is False or generate_memories is False
    if feature_enabled and not explicitly_disabled:
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
    for label, adapter in (("recall.embedder", config.recall_settings().embedder), ("recall.reranker", config.recall_settings().reranker)):
        checks.append((label, not adapter.enabled or adapter.type == "none", "disabled" if not adapter.enabled else f"adapter type {adapter.type}"))
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
