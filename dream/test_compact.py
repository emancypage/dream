import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import compact
from compact import CompactionError, build_prompt, check_output, propose, select_files
from config import DelegateSettings, load_config
from consolidate import MemoryFile

HEAD = "---\nname: topic\n---\n"
SETTINGS = DelegateSettings()


def _file(name, chars):
    body = HEAD + "x" * (chars - len(HEAD))
    return MemoryFile(Path("/memory") / name, body, hashlib.sha256(body.encode()).hexdigest())


def test_select_files_takes_the_longest_topics_and_never_the_index():
    files = [_file("a.md", 41_000), _file("b.md", 55_000), _file("c.md", 45_000),
             _file("MEMORY.md", 59_000), _file("d.md", 40_000)]
    assert [f.path.name for f in select_files(files, SETTINGS)] == ["b.md", "c.md"]


def test_prompt_carries_target_size_and_patches():
    prompt = build_prompt(_file("a.md", 41_000), [{"id": 7, "find": "x", "replacement": "y", "rationale": "r"}], SETTINGS, "")
    assert "25000" in prompt and '"id": 7' in prompt


def test_prompt_does_not_expand_placeholders_inside_the_data():
    entry = MemoryFile(Path("/memory/a.md"), HEAD + "see {{target_chars}} and {{pending_patches}}", "sha")
    prompt = build_prompt(entry, [], SETTINGS, "")
    assert "see {{target_chars}} and {{pending_patches}}" in prompt


@pytest.mark.parametrize("output, message", [
    ({"body": "", "dropped_summary": "", "folded_ids": []}, "empty body"),
    ({"body": HEAD + "x" * 40_001, "dropped_summary": "", "folded_ids": []}, "body longer than compact_over_chars"),
    ({"body": "---\nname: other\n---\nshort", "dropped_summary": "", "folded_ids": []}, "frontmatter name changed"),
    ({"body": HEAD + "short", "dropped_summary": "", "folded_ids": [9]}, "unknown folded id"),
    ({"body": HEAD + "short", "dropped_summary": "", "folded_ids": [7, 7]}, "duplicate folded id"),
])
def test_check_output_refuses(output, message):
    with pytest.raises(CompactionError, match=message):
        check_output(_file("a.md", 41_000), output, {7}, SETTINGS)


def test_propose_uses_the_delegate_stage_and_stops_before_the_provider_on_oversize(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(compact.backend, "generate", lambda stage, *a, **k: calls.append(stage) or SimpleNamespace(
        output={"body": HEAD + "short", "dropped_summary": "old history", "folded_ids": []}, provider="fake", model="m"))
    cfg = load_config(tmp_path / "absent")
    result = propose(_file("a.md", 41_000), [], config=cfg, policy_context="")
    assert calls == ["delegate"] and result.dropped_summary == "old history"
    with pytest.raises(CompactionError, match="input longer than 200000 characters"):
        propose(_file("big.md", 200_001), [], config=cfg, policy_context="")
    assert calls == ["delegate"]
