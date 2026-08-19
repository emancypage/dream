"""Tests for the streaming-status seam in claude_cli (_tool_label, _consume_stream).

These are the brittle bits the live status feature rests on: mapping a tool_use
name (including MCP `mcp__server__tool` names) to a human label, and walking the
stream-json (JSONL) event sequence to fire those labels and capture the final
result. Both are pure — no subprocess — so we feed synthetic event lines.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from claude_cli import _consume_stream, _tool_label  # noqa: E402


def test_tool_label_builtin_tools():
    assert _tool_label("WebSearch") == "🔍 szukam w necie…"
    assert _tool_label("Bash") == "⚙️ odpalam komendę…"
    assert _tool_label("Read") == "📄 czytam pliki…"


def test_tool_label_mcp_parses_server_segment():
    # The exact shape an MCP-backed tool name takes: mcp__<server>__<tool>.
    assert _tool_label("mcp__claude_ai_Gmail__search_threads") == "📧 sprawdzam mail…"
    assert _tool_label("mcp__claude_ai_Google_Calendar__list_events") == "📅 zerkam w kalendarz…"
    assert _tool_label("mcp__claude_ai_Google_Drive__read_file_content") == "📁 grzebię w Drive…"


def test_tool_label_unknown_degrades_gracefully():
    assert _tool_label("SomeFutureTool") == "🛠️ używam narzędzia…"
    assert _tool_label("mcp__unknown_server__do_thing") == "🛠️ używam integracji…"


def _line(obj):
    return json.dumps(obj)


def test_consume_stream_fires_labels_and_returns_result():
    labels = []
    lines = [
        _line({"type": "system", "subtype": "init"}),
        _line({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "name": "WebSearch", "input": {"query": "x"}},
        ]}}),
        _line({"type": "user", "message": {"content": [{"type": "tool_result", "content": "..."}]}}),
        _line({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__claude_ai_Gmail__search_threads", "input": {}},
        ]}}),
        _line({"type": "result", "subtype": "success", "result": "gotowe",
               "usage": {"input_tokens": 3, "output_tokens": 7}, "total_cost_usd": 0.0}),
    ]
    result = _consume_stream(iter(lines), labels.append)

    assert labels == ["🔍 szukam w necie…", "📧 sprawdzam mail…"]
    assert result is not None
    assert result["result"] == "gotowe"
    assert result["usage"]["output_tokens"] == 7


def test_consume_stream_dedups_consecutive_identical_labels():
    labels = []
    lines = [
        _line({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Grep", "input": {}}]}}),
        _line({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Glob", "input": {}}]}}),  # Glob maps to the same label as Grep
        _line({"type": "result", "result": "ok"}),
    ]
    _consume_stream(iter(lines), labels.append)
    assert labels == ["🔎 przeszukuję pliki…"]   # identical consecutive label → fired once


def test_consume_stream_swallows_callback_errors():
    def boom(_label):
        raise RuntimeError("status sink 429")

    lines = [
        _line({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {}}]}}),
        _line({"type": "result", "result": "survived"}),
    ]
    # A raised callback must not abort the walk — we still get the result.
    result = _consume_stream(iter(lines), boom)
    assert result["result"] == "survived"


def test_consume_stream_no_result_event_returns_none():
    lines = [_line({"type": "assistant", "message": {"content": []}})]
    assert _consume_stream(iter(lines), None) is None


def test_consume_stream_ignores_malformed_lines():
    lines = ["not json", "", _line({"type": "result", "result": "ok"})]
    result = _consume_stream(iter(lines), None)
    assert result["result"] == "ok"
