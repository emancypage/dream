"""Tests for consolidate's memory-store prompt budget (read_current_memory)."""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consolidate import read_current_memory  # noqa: E402


def test_no_truncation_under_new_cap():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "big.md").write_text("x" * 200_000, encoding="utf-8")
        blob = read_current_memory(root)
        assert "truncated for prompt budget" not in blob
        assert len(blob) > 200_000


def test_still_truncates_and_warns_over_new_cap():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "big.md").write_text("x" * 900_000, encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            blob = read_current_memory(root)
        assert "truncated for prompt budget" in blob
        assert "truncat" in buf.getvalue().lower()
