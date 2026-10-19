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
        blob = read_current_memory(root, constitution_path=None)
        assert "truncated for prompt budget" not in blob
        assert len(blob) > 200_000


def test_still_truncates_and_warns_over_new_cap():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "big.md").write_text("x" * 900_000, encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            blob = read_current_memory(root, constitution_path=None)
        assert "truncated for prompt budget" in blob
        assert "truncat" in buf.getvalue().lower()


def test_render_memory_file_matches_the_store_format_and_takes_an_annotation():
    from consolidate import render_memory_file

    assert render_memory_file("a.md", "body") == "### FILE: a.md\n\nbody\n"
    assert render_memory_file("a.md", "body", annotation="note") == "### FILE: a.md\nnote\n\nbody\n"


def test_read_current_memory_renders_every_file_through_render_memory_file(monkeypatch):
    import consolidate

    seen = []
    real = consolidate.render_memory_file
    monkeypatch.setattr(consolidate, "render_memory_file",
                        lambda name, body, **kw: seen.append(name) or real(name, body))
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "b.md").write_text("b", encoding="utf-8")
        (root / "a.md").write_text("a", encoding="utf-8")
        blob = read_current_memory(root, constitution_path=None)
    assert seen == ["a.md", "b.md"]
    assert blob == "### FILE: a.md\n\na\n\n---\n### FILE: b.md\n\nb\n"


def test_memory_files_apply_the_read_current_memory_filters():
    import json
    from consolidate import memory_files

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name in ("b.md", "a.md", ".hidden.md", "excluded.md"):
            (root / name).write_text(name, encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "nested.md").write_text("nested", encoding="utf-8")
        (root / ".dream-retention.json").write_text(json.dumps({"memory_paths": ["excluded.md"]}), encoding="utf-8")
        files = memory_files(root, constitution_path=None)
        assert [(path.name, body) for path, body in files] == [("a.md", "a.md"), ("b.md", "b.md")]
        assert all(path.parent == root for path, _body in files)
