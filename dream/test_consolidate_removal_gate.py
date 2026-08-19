"""Tests for consolidate's mechanical mtime gate on `remove` suggestions.

The model proposes `remove` based on content alone (it's never shown file mtimes);
the "untouched for >=N days" half of the staleness criterion is enforced here.
"""
import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consolidate import Suggestion, _drop_fresh_removals, REMOVE_GRACE_DAYS  # noqa: E402


def _touch_days_ago(path: Path, days: int) -> None:
    path.write_text("x", encoding="utf-8")
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).timestamp()
    os.utime(path, (ts, ts))


def test_holds_back_recently_touched_file():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _touch_days_ago(root / "fresh.md", 5)
        sugs = [Suggestion(kind="remove", target_path="fresh.md", body="", rationale="r", source_sessions=[])]
        assert _drop_fresh_removals(sugs, root) == []


def test_keeps_file_past_grace_period():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _touch_days_ago(root / "old.md", REMOVE_GRACE_DAYS + 5)
        sugs = [Suggestion(kind="remove", target_path="old.md", body="", rationale="r", source_sessions=[])]
        kept = _drop_fresh_removals(sugs, root)
        assert [s.target_path for s in kept] == ["old.md"]


def test_ignores_non_remove_kinds_regardless_of_age():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sugs = [Suggestion(kind="update", target_path="whatever.md", body="b", rationale="r", source_sessions=[])]
        assert _drop_fresh_removals(sugs, root) == sugs


def test_remove_of_already_missing_file_passes_through():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sugs = [Suggestion(kind="remove", target_path="gone.md", body="", rationale="r", source_sessions=[])]
        assert _drop_fresh_removals(sugs, root) == sugs
