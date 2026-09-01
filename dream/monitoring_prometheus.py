#!/usr/bin/env python3
"""Idempotently add the Dream metrics exporter to Prometheus."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Sequence


ANCHOR = "  - job_name: personal-rag-workers\n"
JOB_MARKER = "  - job_name: dream\n"
TARGET_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def patch_text(source: str, target: str) -> str:
    if JOB_MARKER in source:
        return source
    if not TARGET_RE.fullmatch(target):
        raise ValueError("Dream exporter target contains unsupported characters")
    if ANCHOR not in source:
        raise ValueError("required anchor job personal-rag-workers is missing")
    job = (
        "  - job_name: dream\n"
        "    scrape_interval: 15s\n"
        "    scrape_timeout: 5s\n"
        "    static_configs:\n"
        f'      - targets: ["{target}"]\n'
        "        labels: {instance: dream}\n\n"
    )
    return source.replace(ANCHOR, job + ANCHOR, 1)


def atomic_write(path: Path, content: str) -> None:
    temporary_name = ""
    mode = path.stat().st_mode
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("target")
    args = parser.parse_args(argv)
    atomic_write(args.path, patch_text(args.path.read_text(encoding="utf-8"), args.target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
