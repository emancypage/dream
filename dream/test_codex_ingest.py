import json
import sqlite3
from pathlib import Path

from dream import open_db
from ingest import ingest_source
from sources.codex_jsonl import CodexJSONLSource


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _meta(source="cli", thread_source="user"):
    return {
        "type": "session_meta",
        "timestamp": "2026-08-08T10:00:00Z",
        "payload": {
            "source": source,
            "thread_source": thread_source,
            "session_id": "session-1",
            "cwd": "/home/user/project",
        },
    }


def _message(role, text, *, item_id, phase=None):
    payload = {
        "type": "message",
        "role": role,
        "id": item_id,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
    }
    if phase:
        payload["phase"] = phase
    return {"type": "response_item", "timestamp": "2026-08-08T10:01:00Z", "payload": payload}


def test_codex_source_keeps_only_real_user_and_final_answers(tmp_path):
    path = tmp_path / "2026" / "08" / "session.jsonl"
    rows = [
        _meta(),
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "id": "bootstrap",
                "content": [
                    {"type": "input_text", "text": "# AGENTS.md instructions for /home/user\n..."},
                    {"type": "input_text", "text": "<environment_context>...</environment_context>"},
                ],
            },
        },
        _message("user", "real request", item_id="u1"),
        _message("assistant", "progress", item_id="a0", phase="commentary"),
        {"type": "response_item", "payload": {"type": "reasoning", "id": "r1"}},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "id": "t1"}},
        _message("assistant", "final result", item_id="a1", phase="final_answer"),
        _message("assistant", "final result", item_id="a1", phase="final_answer"),
    ]
    _write(path, rows)

    source = CodexJSONLSource(tmp_path)
    refs = list(source.discover())
    assert len(refs) == 1
    parsed = source.parse(refs[0])
    assert [(message.role, message.text) for message in parsed.messages] == [
        ("user", "real request"),
        ("assistant", "final result"),
    ]
    assert parsed.project_slug == "-home-user-project"


def test_codex_source_rejects_subagent_and_guardian(tmp_path):
    _write(
        tmp_path / "subagent.jsonl",
        [_meta(source={"subagent": {"thread_spawn": {"depth": 1}}}, thread_source="subagent")],
    )
    _write(
        tmp_path / "guardian.jsonl",
        [_meta(source={"subagent": {"other": "guardian"}}, thread_source="subagent")],
    )
    assert list(CodexJSONLSource(tmp_path).discover()) == []


def test_codex_ingest_is_revision_idempotent(tmp_path):
    transcript = tmp_path / "session.jsonl"
    _write(transcript, [_meta(), _message("user", "one", item_id="u1")])
    conn = open_db(tmp_path / "dream.db")
    source = CodexJSONLSource(tmp_path)

    assert ingest_source(source, conn) == (1, 1, 1)
    assert ingest_source(source, conn) == (1, 0, 0)

    with transcript.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_message("assistant", "two", item_id="a1", phase="final_answer")) + "\n")
    assert ingest_source(source, conn) == (1, 1, 2)
    row = conn.execute(
        "SELECT source, external_session_id, parser_version FROM sessions"
    ).fetchone()
    assert row[0] == "codex"
    assert row[1] == "session-1"
    assert row[2].startswith("codex-jsonl-")
