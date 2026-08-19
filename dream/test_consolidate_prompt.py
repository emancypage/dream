"""Tests for the consolidate prompt template's staleness-audit section."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consolidate import _load_prompt  # noqa: E402


def test_prompt_has_staleness_audit_section():
    assert "# Staleness audit" in _load_prompt()


def test_prompt_names_closure_markers():
    text = _load_prompt()
    assert "ODRZUCONE" in text
    assert "ARCHIWALNE" in text


def test_prompt_tells_model_not_to_judge_age():
    text = _load_prompt()
    assert "Do NOT use file age" in text


def test_prompt_substitution_leaves_no_placeholders():
    filled = (
        _load_prompt()
        .replace("{{today}}", "2026-07-12")
        .replace("{{memory_root}}", "/tmp/memory")
        .replace("{{session_count}}", "0")
        .replace("{{current_memory}}", "(empty)")
        .replace("{{distilled_batch}}", "(empty)")
    )
    assert "{{" not in filled
