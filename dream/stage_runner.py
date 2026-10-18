#!/usr/bin/env python3
"""Run one Dream systemd stage and record its bounded outcome."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

import metrics


STATE_PATH = metrics.state_path_from_env()


def _error_code(return_code: int) -> str:
    if return_code >= 0:
        return f"stage_exit_{abs(return_code)}"
    return f"stage_signal_{abs(return_code)}"


def run_stage(
    stage: str,
    command: Sequence[str],
    *,
    runner: Callable[[Sequence[str]], int] | None = None,
    continue_on_error: bool = False,
) -> int:
    metrics.start_stage(STATE_PATH, stage=stage)
    execute = runner or _run_subprocess
    try:
        return_code = execute(command)
    except OSError:
        metrics.finish_stage(
            STATE_PATH,
            stage=stage,
            success=False,
            error_code="command_not_found",
            continue_on_error=continue_on_error,
        )
        return_code = 127
    else:
        if return_code == 0:
            metrics.finish_stage(STATE_PATH, stage=stage, success=True)
        else:
            metrics.finish_stage(
                STATE_PATH,
                stage=stage,
                success=False,
                error_code=_error_code(return_code),
                continue_on_error=continue_on_error and return_code > 0,
            )
    if continue_on_error and return_code > 0:
        print(f"dream-stage: {stage} exited with status {return_code}; continuing remaining stages. Failure retained in run metrics.",
              file=sys.stderr)
        return 0
    return return_code


def _run_subprocess(command: Sequence[str]) -> int:
    completed = subprocess.run(command, check=False, env=os.environ.copy())
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Run later scheduled stages while retaining this failure in metrics")
    parser.add_argument("stage", choices=metrics.STAGES)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("dream-stage: command is required after --", file=sys.stderr)
        return 2
    return run_stage(args.stage, command, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
