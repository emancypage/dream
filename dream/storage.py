"""SQLite connection modes used by Dream commands."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote


def open_db_readonly(path: Path) -> sqlite3.Connection:
    """Open an existing database without requiring write access to its directory.

    ``immutable=1`` is intentional for sandboxed readers: SQLite does not try
    to create WAL/SHM files or migration backups. The write pipeline continues
    to use Dream's normal opener and remains responsible for schema changes.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.with_name(f"{resolved.name}-wal").exists():
        raise sqlite3.OperationalError("database has active WAL; immutable read deferred")
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn
