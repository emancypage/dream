import json
import sqlite3
from pathlib import Path

from dream import open_db
from ingest import ingest_source
from sources.opencode_sqlite import OpenCodeSQLiteSource, PARSER_VERSION


CREATED_MS = 1_780_000_000_000


def _part(part_id: str, message_id: str, session_id: str, payload: dict, *, seq: int = 0) -> tuple:
    return (
        part_id,
        message_id,
        session_id,
        CREATED_MS + seq,
        CREATED_MS + seq,
        json.dumps(payload),
    )


def _message(message_id: str, session_id: str, role: str, *, seq: int = 0, extra: dict | None = None) -> tuple:
    payload = {"role": role, "time": {"created": CREATED_MS + seq}, "agent": "build"}
    if extra:
        payload.update(extra)
    return (message_id, session_id, CREATED_MS + seq, CREATED_MS + seq, json.dumps(payload))


def _session(
    session_id: str,
    *,
    parent_id: str | None = None,
    directory: str = "/home/user/project",
    updated: int = CREATED_MS,
) -> tuple:
    return (
        session_id,
        "proj-1",
        None,
        parent_id,
        "slug",
        directory,
        None,
        "title",
        "1.18.20",
        0.0,
        0,
        0,
        0,
        0,
        0,
        "build",
        None,
        CREATED_MS,
        updated,
    )


def make_fixture(path: Path) -> Path:
    db_path = path / "opencode.db" if path.is_dir() else path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE project (
            id TEXT PRIMARY KEY,
            worktree TEXT NOT NULL,
            time_created INTEGER NOT NULL,
            time_updated INTEGER NOT NULL,
            sandboxes TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE workspace (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            branch TEXT,
            directory TEXT,
            extra TEXT,
            project_id TEXT NOT NULL,
            time_used INTEGER NOT NULL
        );
        CREATE TABLE session (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            workspace_id TEXT,
            parent_id TEXT,
            slug TEXT NOT NULL,
            directory TEXT NOT NULL,
            path TEXT,
            title TEXT NOT NULL,
            version TEXT NOT NULL,
            cost REAL NOT NULL DEFAULT 0,
            tokens_input INTEGER NOT NULL DEFAULT 0,
            tokens_output INTEGER NOT NULL DEFAULT 0,
            tokens_reasoning INTEGER NOT NULL DEFAULT 0,
            tokens_cache_read INTEGER NOT NULL DEFAULT 0,
            tokens_cache_write INTEGER NOT NULL DEFAULT 0,
            agent TEXT,
            model TEXT,
            time_created INTEGER NOT NULL,
            time_updated INTEGER NOT NULL
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL,
            time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL,
            time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO project(id, worktree, time_created, time_updated) VALUES ('proj-1', '/home/user/project', ?, ?)",
        (CREATED_MS, CREATED_MS),
    )
    conn.execute(
        """
        INSERT INTO session(
            id, project_id, workspace_id, parent_id, slug, directory, path, title, version,
            cost, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write,
            agent, model, time_created, time_updated
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        _session("ses_root"),
    )
    conn.execute(
        """
        INSERT INTO session(
            id, project_id, workspace_id, parent_id, slug, directory, path, title, version,
            cost, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write,
            agent, model, time_created, time_updated
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        _session("ses_child", parent_id="ses_root", directory="/home/user/project"),
    )
    conn.executemany(
        "INSERT INTO message(id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?)",
        [
            _message("msg_u", "ses_root", "user", seq=1),
            _message("msg_a", "ses_root", "assistant", seq=2),
            _message("msg_child", "ses_child", "user", seq=1),
        ],
    )
    conn.executemany(
        "INSERT INTO part(id, message_id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?,?)",
        [
            _part("p_u", "msg_u", "ses_root", {"type": "text", "text": "real request"}, seq=1),
            _part(
                "p_syn",
                "msg_u",
                "ses_root",
                {"type": "text", "text": "injected prompt", "synthetic": True},
                seq=0,
            ),
            _part("p_file", "msg_u", "ses_root", {"type": "file", "filename": "note.md"}, seq=3),
            _part("p_think", "msg_a", "ses_root", {"type": "reasoning", "text": "hidden chain"}, seq=4),
            _part(
                "p_tool",
                "msg_a",
                "ses_root",
                {"type": "tool", "tool": "bash", "state": {"status": "completed", "output": "secret dump"}},
                seq=5,
            ),
            _part("p_a", "msg_a", "ses_root", {"type": "text", "text": "final result"}, seq=6),
            _part("p_child", "msg_child", "ses_child", {"type": "text", "text": "subagent ask"}, seq=1),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_opencode_reads_v2_workspace_schema_without_branch(tmp_path):
    db_path = make_fixture(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            DROP TABLE workspace;
            CREATE TABLE workspace (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                binding TEXT,
                created_at INTEGER NOT NULL,
                last_used_at INTEGER NOT NULL
            );
        """)
    source = OpenCodeSQLiteSource(tmp_path)
    parsed = source.parse(next(iter(source.discover())))
    assert parsed.git_branch is None
    assert [(message.role, message.text) for message in parsed.messages] == [
        ("user", "real request"),
        ("assistant", "final result"),
    ]


def test_opencode_preserves_branch_from_legacy_workspace(tmp_path):
    db_path = make_fixture(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            INSERT INTO workspace(id, type, branch, project_id, time_used)
            VALUES ('workspace-1', 'worktree', 'topic', 'proj-1', ?)
        """, (CREATED_MS,))
        conn.execute("UPDATE session SET workspace_id='workspace-1' WHERE id='ses_root'")
    source = OpenCodeSQLiteSource(tmp_path)
    assert source.parse(next(iter(source.discover()))).git_branch == "topic"


def test_opencode_reads_sessions_without_workspace_id(tmp_path):
    db_path = make_fixture(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE session DROP COLUMN workspace_id")
    source = OpenCodeSQLiteSource(tmp_path)
    parsed = source.parse(next(iter(source.discover())))
    assert parsed.git_branch is None
    assert len(parsed.messages) == 2


def test_opencode_source_keeps_only_visible_user_and_assistant_text(tmp_path):
    db_path = make_fixture(tmp_path)
    source = OpenCodeSQLiteSource(tmp_path)
    refs = list(source.discover())
    assert [ref.external_session_id for ref in refs] == ["ses_root"]
    parsed = source.parse(refs[0])
    assert [(message.role, message.text) for message in parsed.messages] == [
        ("user", "real request"),
        ("assistant", "final result"),
    ]
    assert parsed.source == "opencode"
    assert parsed.project_slug == "-home-user-project"
    assert parsed.cwd == "/home/user/project"
    assert parsed.parser_version == PARSER_VERSION
    assert parsed.path == db_path
    assert parsed.started_at.endswith("Z")


def test_opencode_source_missing_db_discovers_nothing(tmp_path):
    assert list(OpenCodeSQLiteSource(tmp_path).discover()) == []


def test_opencode_source_accepts_db_file_root(tmp_path):
    db_path = make_fixture(tmp_path)
    refs = list(OpenCodeSQLiteSource(db_path).discover())
    assert len(refs) == 1
    assert refs[0].path == db_path


def test_opencode_source_opens_read_only(tmp_path, monkeypatch):
    make_fixture(tmp_path)
    seen: list[str] = []
    real_connect = sqlite3.connect

    def wrapped(dsn, *args, **kwargs):
        seen.append(str(dsn))
        return real_connect(dsn, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", wrapped)
    source = OpenCodeSQLiteSource(tmp_path)
    refs = list(source.discover())
    parsed = source.parse(refs[0])
    assert parsed is not None
    assert seen
    assert all("mode=ro" in dsn for dsn in seen)


def test_opencode_ingest_is_revision_idempotent(tmp_path):
    db_path = make_fixture(tmp_path)
    conn = open_db(tmp_path / "dream.db")
    source = OpenCodeSQLiteSource(tmp_path)
    digest_before = db_path.read_bytes()

    assert ingest_source(source, conn) == (1, 1, 2)
    assert ingest_source(source, conn) == (1, 0, 0)
    assert db_path.read_bytes() == digest_before

    src = sqlite3.connect(db_path)
    src.execute(
        "INSERT INTO message(id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?)",
        _message("msg_u2", "ses_root", "user", seq=10),
    )
    src.execute(
        "INSERT INTO part(id, message_id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?,?)",
        _part("p_u2", "msg_u2", "ses_root", {"type": "text", "text": "follow up"}, seq=10),
    )
    src.execute("UPDATE session SET time_updated=? WHERE id='ses_root'", (CREATED_MS + 10,))
    src.commit()
    src.close()

    source = OpenCodeSQLiteSource(tmp_path)
    assert ingest_source(source, conn) == (1, 1, 3)
    row = conn.execute(
        "SELECT source, external_session_id, parser_version FROM sessions"
    ).fetchone()
    assert row == ("opencode", "ses_root", PARSER_VERSION)
