from __future__ import annotations

import hashlib
from pathlib import Path


def file_revision(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_slug(cwd: str | None, fallback: str) -> str:
    return cwd.replace("/", "-") if cwd else fallback
