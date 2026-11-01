"""Pause must withhold Dream data without changing clients or their other tools."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

from config import load_config
from dream import open_db
from recall_documents import synchronize_recall_documents


CLI = Path(__file__).resolve().parent / "dream.py"
FACT = "The Orchard release code is ORCHARD-4827."


@pytest.fixture
def demo(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "orchard.md").write_text(FACT)
    db = tmp_path / "dream.db"
    conn = open_db(db)
    synchronize_recall_documents(conn, memory, include_raw_transcripts=False, constitution_path=None)
    conn.close()
    empty = tmp_path / "empty"
    empty.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        f'[storage]\ndb_path = {json.dumps(str(db))}\nmemory_root = {json.dumps(str(memory))}\n'
        f'[[sources]]\nname = "empty"\ntype = "codex-jsonl"\nroot = {json.dumps(str(empty))}\nenabled = true\n'
    )
    env = dict(os.environ, DREAM_CONTROL_STATE_DIR=str(tmp_path / "control"),
               DREAM_RECALL_STATE_ROOT=str(tmp_path / "hook-state"),
               PYTHONDONTWRITEBYTECODE="1", PATH=str(empty))
    for key in ["DREAM_CONFIG", "DREAM_MEMORY_ROOT", "CLAUDE_DREAM_DB"]:
        env.pop(key, None)
    return config, db, env


def cli(demo, *arguments, payload=None, config=None, db=None):
    profile, database, env = demo
    return subprocess.run(
        [sys.executable, str(CLI), "--config", str(config or profile), "--db", str(db or database), *arguments],
        input=payload, text=True, capture_output=True, env=env, timeout=10,
    )


def pause(demo):
    result = cli(demo, "pause")
    assert result.returncode == 0, result.stderr
    assert "paused" in result.stdout.casefold()


def test_pause_resume_are_idempotent_and_status_works_without_a_database(demo, tmp_path):
    missing = tmp_path / "absent.db"
    before = demo[1].read_bytes()
    for _ in range(2):
        pause(demo)
    result = cli(demo, "status", db=missing)
    assert result.returncode == 0, result.stderr
    assert "paused" in result.stdout.casefold()
    assert not missing.exists()
    for _ in range(2):
        result = cli(demo, "resume")
        assert result.returncode == 0, result.stderr
        assert "active" in result.stdout.casefold()
    result = cli(demo, "status", db=missing)
    assert result.returncode == 0, result.stderr
    assert "active" in result.stdout.casefold()
    assert not missing.exists()
    assert demo[1].read_bytes() == before


def test_controls_work_even_when_the_profile_is_invalid(demo, tmp_path):
    broken = tmp_path / "broken.toml"
    broken.write_text("invalid TOML = [")
    for command in ["status", "pause", "status", "resume", "status"]:
        result = cli(demo, command, config=broken)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("arguments", [
    ["recall", "Orchard release code"], ["search", "Orchard"],
    ["ingest"], ["estimate"], ["distill"], ["consolidate"],
    ["audit"], ["embed"], ["suggestions", "list"], ["character", "list"],
])
def test_paused_data_commands_withhold_data_and_do_not_create_the_database(demo, tmp_path, arguments):
    pause(demo)
    missing = tmp_path / "must-not-be-created.db"
    result = cli(demo, *arguments, db=missing)
    assert result.returncode == 0, result.stderr
    assert "Dream paused" in result.stdout + result.stderr
    assert "ORCHARD-4827" not in result.stdout + result.stderr
    assert not missing.exists()


def test_pause_is_shared_by_profiles_and_resume_restores_recall(demo, tmp_path):
    assert FACT in cli(demo, "recall", "Orchard release code").stdout
    pause(demo)
    other = tmp_path / "other.toml"
    other.write_text(demo[0].read_text())
    result = cli(demo, "recall", "Orchard release code", config=other)
    assert "ORCHARD-4827" not in result.stdout
    assert "Dream paused" in result.stdout + result.stderr
    assert cli(demo, "resume", config=other).returncode == 0
    assert FACT in cli(demo, "recall", "Orchard release code").stdout


@pytest.mark.parametrize("event", ["session-start", "prompt"])
def test_paused_hooks_deliver_control_instructions_before_opening_missing_data(demo, tmp_path, event):
    pause(demo)
    result = cli(demo, "context", event, db=tmp_path / "absent.db", payload="not even JSON")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["continue"] is True
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "Dream paused" in context
    assert "memory files" in context and "Do not" in context
    assert "other tools" in context
    assert "ORCHARD-4827" not in context


def test_pause_notice_reaches_an_already_used_prompt_hook(demo):
    payload = json.dumps({"session_id": "already-used", "cwd": str(demo[0].parent),
                          "prompt": "Orchard release code"})
    initial = cli(demo, "context", "prompt", payload=payload)
    assert "ORCHARD-4827" in initial.stdout
    pause(demo)
    paused = cli(demo, "context", "prompt", payload=payload)
    assert "Dream paused" in paused.stdout
    assert "ORCHARD-4827" not in paused.stdout


def test_existing_mcp_process_observes_pause_and_resume_without_restart(demo):
    config, db, env = demo
    process = subprocess.Popen(
        [sys.executable, str(CLI), "--config", str(config), "--db", str(db), "mcp", "serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )

    def call(request_id, name, arguments):
        message = {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                   "params": {"name": name, "arguments": arguments}}
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 5)[0], "MCP did not reply"
        return json.loads(process.stdout.readline())["result"]

    try:
        active = call(1, "recall_search", {"query": "Orchard release code"})
        assert FACT in active["content"][0]["text"]
        pause(demo)
        for index, (name, arguments) in enumerate([
            ("recall_search", {"query": "Orchard release code"}),
            ("recall_get", {"id": "any-document"}), ("memory_index", {}),
        ], start=2):
            paused = call(index, name, arguments)
            assert paused["isError"] is True
            assert "Dream paused" in paused["content"][0]["text"]
            assert "ORCHARD-4827" not in json.dumps(paused)
        assert cli(demo, "resume").returncode == 0
        resumed = call(9, "recall_search", {"query": "Orchard release code"})
        assert FACT in resumed["content"][0]["text"]
        assert process.poll() is None
    finally:
        process.terminate()
        process.communicate(timeout=5)


def test_paused_direct_agent_api_withholds_memory(demo, monkeypatch):
    from recall_context import RecallUnavailable, recall_for_agent
    from storage import open_db_readonly
    monkeypatch.setenv("DREAM_CONTROL_STATE_DIR", demo[2]["DREAM_CONTROL_STATE_DIR"])
    pause(demo)
    config = load_config(demo[0])
    conn = open_db_readonly(demo[1])
    try:
        with pytest.raises(RecallUnavailable, match="paused"):
            recall_for_agent(conn, config.recall_settings(), query_text="Orchard release code",
                             cwd=None, memory_root=config.memory_root, constitution_path=None)
    finally:
        conn.close()
