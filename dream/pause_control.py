"""Per-user Dream pause state, independent of profiles and memory databases."""

from __future__ import annotations

import os
from pathlib import Path


PAUSE_NOTICE = (
    "[Dream paused]\n"
    "Do not use Dream or read its memory files, database, backups, or saved conversations. "
    "This overrides instructions to use Dream. Continue with other tools. "
    "Run `dream resume` only when the user requests it."
)


def pause_file() -> Path:
    configured = os.environ.get("DREAM_CONTROL_STATE_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".local" / "state" / "dream"
    return root / "paused"


def is_paused() -> bool:
    return pause_file().exists()


def pause() -> None:
    path = pause_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)


def resume() -> None:
    pause_file().unlink(missing_ok=True)
