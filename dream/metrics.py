#!/usr/bin/env python3
"""Persist and expose the status of the scheduled Dream pipeline.

The exporter intentionally exposes only bounded labels and numeric values.  It
never sends the error text from a provider or transcript to Prometheus.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Mapping, Sequence


DEFAULT_STATE_PATH = Path.home() / ".local" / "state" / "dream" / "run.json"
DEFAULT_STALE_AFTER = 36 * 60 * 60
STAGES = ("ingest", "distill", "consolidate", "curation", "character_generate", "character_judge")
STATUS_VALUES = ("never", "running", "success", "failed", "stale")
SAFE_LABEL = re.compile(r"^[a-zA-Z0-9_.:-]+$")


def state_path_from_env() -> Path:
    return Path(os.environ.get("DREAM_METRICS_STATE", str(DEFAULT_STATE_PATH))).expanduser()


def stale_after_from_env() -> float:
    raw = os.environ.get("DREAM_METRICS_STALE_AFTER_SECONDS")
    if raw is None:
        return float(DEFAULT_STALE_AFTER)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("DREAM_METRICS_STALE_AFTER_SECONDS must be numeric") from exc
    if value <= 0:
        raise ValueError("DREAM_METRICS_STALE_AFTER_SECONDS must be positive")
    return value


def _now(value: float | None) -> float:
    return time.time() if value is None else float(value)


def _read_state(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, state: Mapping[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    content = json.dumps(dict(state), ensure_ascii=True, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return dict(state)


def _duration(state: Mapping[str, object], finished_at: float) -> float:
    started = state.get("started_at")
    try:
        return max(0.0, finished_at - float(started))
    except (TypeError, ValueError):
        return 0.0


def start_run(path: Path, *, now: float | None = None) -> dict[str, object]:
    timestamp = _now(now)
    return _write_state(
        path,
        {
            "schema_version": 1,
            "status": "running",
            "stage": "starting",
            "started_at": timestamp,
            "completed_stages": [],
        },
    )


def start_stage(path: Path, *, stage: str, now: float | None = None) -> dict[str, object]:
    if stage not in STAGES:
        raise ValueError(f"unknown Dream stage: {stage}")
    state = _read_state(path)
    if state.get("status") != "running":
        state = start_run(path, now=now)
    state["status"] = "running"
    state["stage"] = stage
    state.pop("error_code", None)
    return _write_state(path, state)


def finish_stage(
    path: Path,
    *,
    stage: str,
    success: bool,
    now: float | None = None,
    error_code: str | None = None,
    continue_on_error: bool = False,
) -> dict[str, object]:
    if stage not in STAGES:
        raise ValueError(f"unknown Dream stage: {stage}")
    timestamp = _now(now)
    state = _read_state(path)
    state["stage"] = stage
    if success:
        completed = state.get("completed_stages", [])
        if not isinstance(completed, list):
            completed = []
        if stage not in completed:
            completed.append(stage)
        state["completed_stages"] = completed
        state["status"] = "running"
        state.pop("error_code", None)
        return _write_state(path, state)

    state["error_code"] = _safe_label(error_code or "stage_failed")
    failures = state.get("failed_stages", [])
    if not isinstance(failures, list):
        failures = []
    failures.append({"stage": stage, "error_code": state["error_code"]})
    state["failed_stages"] = failures
    state["status"] = "running" if continue_on_error else "failed"
    if not continue_on_error:
        state["finished_at"] = timestamp
        state["duration_seconds"] = _duration(state, timestamp)
    return _write_state(path, state)


def finish_run(
    path: Path,
    *,
    success: bool,
    now: float | None = None,
    error_code: str | None = None,
) -> dict[str, object]:
    timestamp = _now(now)
    state = _read_state(path)
    state["finished_at"] = timestamp
    state["duration_seconds"] = _duration(state, timestamp)
    failures = state.get("failed_stages", [])
    if success and failures:
        state["status"] = "failed"
        state["stage"] = failures[0]["stage"]
        state["error_code"] = failures[0]["error_code"]
    elif success:
        state["status"] = "success"
        state["stage"] = STAGES[-1]
        state["completed_stages"] = list(STAGES)
        state.pop("error_code", None)
    else:
        state["status"] = "failed"
        state["error_code"] = _safe_label(error_code or "run_failed")
    return _write_state(path, state)


def seed_success(
    path: Path,
    *,
    started_at: float,
    finished_at: float,
    source: str = "import",
) -> dict[str, object]:
    """Record a success verified by an external authoritative run ledger."""
    started = float(started_at)
    finished = float(finished_at)
    if finished < started:
        raise ValueError("finished_at must not precede started_at")
    return _write_state(
        path,
        {
            "schema_version": 1,
            "status": "success",
            "stage": STAGES[-1],
            "source": _safe_label(source),
            "started_at": started,
            "finished_at": finished,
            "duration_seconds": finished - started,
            "completed_stages": list(STAGES),
        },
    )


def finalize_run(
    path: Path,
    *,
    service_result: str,
    exit_status: str,
    now: float | None = None,
) -> dict[str, object]:
    state = _read_state(path)
    if state.get("status") == "success" and service_result in ("", "success"):
        return state
    if state.get("status") == "failed":
        return state
    if service_result in ("", "success") and state.get("status") == "never":
        return state
    result = _safe_label(service_result or "unknown")
    status = _safe_label(exit_status or "unknown")
    return finish_run(
        path,
        success=False,
        now=now,
        error_code=f"service_{result}_{status}",
    )


def _safe_label(value: object) -> str:
    text = str(value or "unknown")
    if SAFE_LABEL.fullmatch(text):
        return text[:80]
    return "unknown"


def health_status(state: Mapping[str, object], *, now: float, stale_after: float) -> str:
    status = state.get("status")
    if status == "running":
        return "running"
    if status == "failed":
        return "failed"
    if status == "success":
        try:
            finished = float(state["finished_at"])
        except (KeyError, TypeError, ValueError):
            return "stale"
        return "stale" if now - finished > stale_after else "success"
    return "never"


def health_value(state: Mapping[str, object], *, now: float, stale_after: float) -> int:
    values = {"never": -1, "failed": 0, "stale": 1, "running": 2, "success": 3}
    return values[health_status(state, now=now, stale_after=stale_after)]


def _label(value: object) -> str:
    text = str(value or "unknown")
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value: object, default: float = 0.0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def render_metrics(
    state: Mapping[str, object],
    *,
    now: float | None = None,
    stale_after: float = DEFAULT_STALE_AFTER,
) -> str:
    timestamp = _now(now)
    status = health_status(state, now=timestamp, stale_after=stale_after)
    health = health_value(state, now=timestamp, stale_after=stale_after)
    stage = _safe_label(state.get("stage", "unknown"))
    error_code = _safe_label(state.get("error_code", "none"))
    lines = [
        "# HELP dream_run_health Current Dream pipeline health: "
        "-1 never, 0 failed, 1 stale, 2 running, 3 success.",
        "# TYPE dream_run_health gauge",
        f"dream_run_health {health}",
        "# HELP dream_run_status One-hot current Dream pipeline status.",
        "# TYPE dream_run_status gauge",
    ]
    for value in STATUS_VALUES:
        lines.append(f'dream_run_status{{status="{value}"}} {1 if status == value else 0}')
    lines.extend(
        [
            "# HELP dream_last_run_timestamp_seconds Unix timestamp of the last "
            "completed Dream run.",
            "# TYPE dream_last_run_timestamp_seconds gauge",
            f"dream_last_run_timestamp_seconds {_number(state.get('finished_at'))}",
            "# HELP dream_last_run_duration_seconds Duration of the last completed Dream run.",
            "# TYPE dream_last_run_duration_seconds gauge",
            f"dream_last_run_duration_seconds {_number(state.get('duration_seconds'))}",
            "# HELP dream_last_run_info Bounded metadata for the last Dream run.",
            "# TYPE dream_last_run_info gauge",
            f'dream_last_run_info{{result="{_label(status)}",'
            f'stage="{_label(stage)}",error_code="{_label(error_code)}"}} 1',
            "# HELP dream_last_run_started_timestamp_seconds Unix timestamp when "
            "the current or last Dream run started.",
            "# TYPE dream_last_run_started_timestamp_seconds gauge",
            f"dream_last_run_started_timestamp_seconds {_number(state.get('started_at'))}",
        ]
    )
    return "\n".join(lines) + "\n"


def serve_metrics(
    *,
    state_path: Path,
    bind: str,
    port: int,
    stale_after: float,
) -> None:
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = collect_metrics(
                state_path,
                stale_after=stale_after,
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    with ThreadingHTTPServer((bind, port), MetricsHandler) as server:
        server.serve_forever()


def _recall_usage_lines(conn: sqlite3.Connection) -> list[str]:
    """The unused-memory gauge: a bare count, never a document id or path.

    Omitted while usage is unknown (database not migrated, no marker harvested
    yet, unreadable recall tables) and while the observation window is shorter
    than the 90 days the gauge counts, so a dashboard never reads a partial
    window as "unused for 90 days".
    """
    from recall_usage import UNUSED_DAYS, usage_report
    try:
        report = usage_report(conn)
    except (ValueError, sqlite3.Error):
        return []
    if report.state != "observed" or report.window_shorter_than_days:
        return []
    return [
        f"# HELP dream_recall_documents_unused_90d Approved memory documents automatic recall "
        f"has not delivered in the last {UNUSED_DAYS} days; only exported once usage has been "
        f"observed for at least {UNUSED_DAYS} days.",
        "# TYPE dream_recall_documents_unused_90d gauge",
        f"dream_recall_documents_unused_90d {len(report.unused)}",
    ]


def collect_metrics(state_path: Path, *, config=None, stale_after: float = DEFAULT_STALE_AFTER) -> str:
    """Read the review queue on every scrape, independently of nightly run state."""
    from config import load_config
    from storage import open_db_readonly
    from character.short import pending_review_count
    text = render_metrics(_read_state(state_path), stale_after=stale_after)
    lines = ['# HELP dream_character_short_review_source_up Whether the short review queue was readable.',
             '# TYPE dream_character_short_review_source_up gauge']
    conn = None
    try:
        try:
            cfg = config if config is not None else load_config()
            conn = open_db_readonly(cfg.db_path)
            count = pending_review_count(conn, cfg)
        except (OSError, ValueError, sqlite3.Error):
            lines.append('dream_character_short_review_source_up 0')
        else:
            lines.extend(['dream_character_short_review_source_up 1',
                          '# HELP dream_character_short_drafts_pending Current short example drafts awaiting human review.',
                          '# TYPE dream_character_short_drafts_pending gauge',
                          f'dream_character_short_drafts_pending {count}'])
        # Independent of the review queue: either can be readable without the other.
        if conn is not None:
            lines.extend(_recall_usage_lines(conn))
    finally:
        if conn is not None:
            conn.close()
    return text + '\n'.join(lines) + '\n'


def _command(args: argparse.Namespace) -> int:
    path = state_path_from_env()
    if args.action == "start":
        start_run(path)
    elif args.action == "stage-start":
        start_stage(path, stage=args.stage)
    elif args.action == "stage-success":
        finish_stage(path, stage=args.stage, success=True)
    elif args.action == "stage-failure":
        finish_stage(path, stage=args.stage, success=False, error_code=args.error_code)
    elif args.action == "success":
        finish_run(path, success=True)
    elif args.action == "seed-success":
        seed_success(
            path,
            started_at=args.started_at,
            finished_at=args.finished_at,
            source=args.source,
        )
    elif args.action == "finalize":
        finalize_run(path, service_result=args.service_result, exit_status=args.exit_status)
    elif args.action == "serve":
        serve_metrics(
            state_path=path,
            bind=args.bind,
            port=args.port,
            stale_after=stale_after_from_env(),
        )
    else:
        raise ValueError(f"unknown metrics action: {args.action}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("start")
    stage_start = sub.add_parser("stage-start")
    stage_start.add_argument("stage", choices=STAGES)
    stage_success = sub.add_parser("stage-success")
    stage_success.add_argument("stage", choices=STAGES)
    stage_failure = sub.add_parser("stage-failure")
    stage_failure.add_argument("stage", choices=STAGES)
    stage_failure.add_argument("error_code")
    sub.add_parser("success")
    seed = sub.add_parser("seed-success")
    seed.add_argument("started_at", type=float)
    seed.add_argument("finished_at", type=float)
    seed.add_argument("--source", default="import")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("service_result")
    finalize.add_argument("exit_status")
    serve = sub.add_parser("serve")
    serve.add_argument("--bind", default=os.environ.get("DREAM_METRICS_BIND", "127.0.0.1"))
    serve.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DREAM_METRICS_PORT", "9409")),
    )
    try:
        return _command(parser.parse_args(argv))
    except (OSError, ValueError) as exc:
        print(f"dream-metrics: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
