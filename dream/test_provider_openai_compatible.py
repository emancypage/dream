"""OpenAI-compatible generation provider against a loopback fake server."""

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import backend
from config import ConfigError, load_config
from distill import DISTILL_SCHEMA, distill_session
from model_types import GenerationRequest, ProviderError, SchemaValidationError
from providers import openai_compatible
from providers.registry import create_provider, preflight_provider


ROOT = Path(__file__).parent
SCHEMA = (ROOT / "schema.sql").read_text(encoding="utf-8")
VALUE_SCHEMA = {
    "type": "object",
    "required": ["value"],
    "additionalProperties": False,
    "properties": {"value": {"type": "string"}},
}
DISTILL_OUTPUT = {
    "summary": "Configured the local model server.",
    "user_facts": [],
    "preferences": ["Prefers local models for distillation."],
    "projects": [],
    "references": [],
    "decisions": [],
    "open_threads": [],
}


class _FakeServer:
    """Loopback OpenAI-compatible server; each test sets `responder`."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responder = lambda request: (200, {"object": "list", "data": []})
        self.release = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                request = {
                    "method": self.command,
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": json.loads(raw) if raw else None,
                }
                fake.requests.append(request)
                status, body, *extra = fake.responder(request)
                headers = extra[0] if extra else {}
                payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass  # The client gave up (timeout test).

            do_GET = _handle
            do_POST = _handle

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def chat_requests(self) -> list[dict]:
        return [request for request in self.requests if request["path"] == "/v1/chat/completions"]


@pytest.fixture
def server():
    fake = _FakeServer()
    yield fake
    fake.close()


def _completion(content, *, model="qwen3-32b-q4_k_m.gguf", usage=None):
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def _request(**overrides) -> GenerationRequest:
    values = {
        "system_prompt": "Return JSON.",
        "prompt": "Say ok.",
        "schema": VALUE_SCHEMA,
        "model": None,
        "timeout_seconds": 30,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _settings(server, **overrides) -> dict:
    settings = {"type": "openai-compatible", "base_url": server.base_url, "model": "qwen3-32b"}
    settings.update(overrides)
    return settings


def test_schema_strict_success_sends_json_schema_and_maps_usage(server):
    server.responder = lambda request: (200, _completion(
        json.dumps({"value": "ok"}), usage={"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127},
    ))

    result = openai_compatible.generate(_request(), _settings(server))

    assert result.output == {"value": "ok"}
    assert result.provider == "openai-compatible"
    assert result.input_tokens == 120 and result.output_tokens == 7
    [request] = server.requests
    assert request["method"] == "POST" and request["path"] == "/v1/chat/completions"
    body = request["body"]
    assert body["model"] == "qwen3-32b"
    assert body["stream"] is False
    assert "tools" not in body
    assert body["messages"] == [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Say ok."},
    ]
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == VALUE_SCHEMA
    assert response_format["json_schema"]["strict"] is True


def test_provenance_records_server_reported_model_not_configured_alias(server):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"}), model="served/qwen3-32b-Q4"))

    result = openai_compatible.generate(_request(), _settings(server))

    assert result.model == "served/qwen3-32b-Q4"
    assert result.usage is None


def test_stage_model_overrides_provider_model(server):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"})))

    openai_compatible.generate(_request(model="stage-model"), _settings(server))

    assert server.requests[0]["body"]["model"] == "stage-model"


def test_request_without_any_model_omits_model_field(server):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"}), model="loaded.gguf"))
    settings = _settings(server)
    del settings["model"]

    result = openai_compatible.generate(_request(), settings)

    assert "model" not in server.requests[0]["body"]
    assert result.model == "loaded.gguf"


def test_base_url_with_v1_suffix_is_not_doubled(server):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"})))

    openai_compatible.generate(_request(), _settings(server, base_url=server.base_url + "/v1/"))

    assert server.requests[0]["path"] == "/v1/chat/completions"


def test_rejected_json_schema_falls_back_once_to_json_object(server):
    def responder(request):
        if request["body"]["response_format"]["type"] == "json_schema":
            return 400, {"error": {"message": "response_format type json_schema is not supported", "type": "invalid_request_error"}}
        return 200, _completion(json.dumps({"value": "fallback"}))

    server.responder = responder

    result = openai_compatible.generate(_request(), _settings(server))

    assert result.output == {"value": "fallback"}
    first, second = server.chat_requests()
    assert first["body"]["response_format"]["type"] == "json_schema"
    assert second["body"]["response_format"] == {"type": "json_object"}
    # json_object mode cannot enforce the shape, so the schema travels in the prompt.
    assert json.dumps(VALUE_SCHEMA, sort_keys=True) in second["body"]["messages"][0]["content"]
    assert second["body"]["messages"][1] == {"role": "user", "content": "Say ok."}


def test_error_body_naming_response_format_triggers_fallback(server):
    def responder(request):
        if request["body"]["response_format"]["type"] == "json_schema":
            return 500, {"error": {"message": "Unsupported response_format"}}
        return 200, _completion(json.dumps({"value": "fallback"}))

    server.responder = responder

    assert openai_compatible.generate(_request(), _settings(server)).output == {"value": "fallback"}
    assert len(server.chat_requests()) == 2


def test_fallback_output_is_validated_host_side(server):
    def responder(request):
        if request["body"]["response_format"]["type"] == "json_schema":
            return 422, {"error": "unprocessable response_format"}
        return 200, _completion(json.dumps({"value": 3}))

    server.responder = responder

    with pytest.raises(SchemaValidationError):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 2


def test_fallback_is_attempted_exactly_once(server):
    server.responder = lambda request: (400, {"error": {"message": "bad response_format"}})

    with pytest.raises(ProviderError, match="400"):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 2


@pytest.mark.parametrize("status", [401, 404, 500, 503])
def test_other_http_errors_are_never_retried(server, status):
    server.responder = lambda request: (status, {"error": {"message": "server unhappy"}})

    with pytest.raises(ProviderError, match=str(status)):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 1


def test_invalid_json_content_is_rejected_without_retry(server):
    server.responder = lambda request: (200, _completion("this is not json"))

    with pytest.raises(ProviderError, match="JSON"):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 1


def test_schema_mismatch_in_strict_mode_is_rejected_without_retry(server):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok", "extra": 1})))

    with pytest.raises(SchemaValidationError):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 1


def test_fenced_json_content_is_accepted(server):
    server.responder = lambda request: (200, _completion('```json\n{"value": "ok"}\n```'))

    assert openai_compatible.generate(_request(), _settings(server)).output == {"value": "ok"}


@pytest.mark.parametrize("body", [
    {"object": "chat.completion", "choices": []},
    _completion(None),
    b"<html>not json</html>",
])
def test_malformed_responses_are_provider_errors(server, body):
    server.responder = lambda request: (200, body)

    with pytest.raises(ProviderError):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 1


def test_timeout_is_a_provider_error_without_retry(server):
    def responder(request):
        server.release.wait(5)
        return 200, _completion(json.dumps({"value": "late"}))

    server.responder = responder

    with pytest.raises(ProviderError, match="timed out"):
        openai_compatible.generate(_request(timeout_seconds=1), _settings(server))
    assert len(server.chat_requests()) == 1


def test_provider_timeout_is_only_a_fallback_for_direct_requests(server):
    def responder(request):
        server.release.wait(5)
        return 200, _completion(json.dumps({"value": "late"}))

    server.responder = responder

    with pytest.raises(ProviderError, match="timed out after 1s"):
        openai_compatible.generate(_request(timeout_seconds=0), _settings(server, timeout_seconds=1))


def test_connection_refused_is_a_provider_error(server):
    url = server.base_url
    server.close()

    with pytest.raises(ProviderError):
        openai_compatible.generate(_request(), _settings(server, base_url=url))


def test_redirects_are_not_followed(server):
    server.responder = lambda request: (302, {}, {"Location": server.base_url + "/elsewhere"})

    with pytest.raises(ProviderError):
        openai_compatible.generate(_request(), _settings(server))
    assert [request["path"] for request in server.requests] == ["/v1/chat/completions"]


def test_environment_proxy_is_ignored(server, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"})))

    assert openai_compatible.generate(_request(), _settings(server)).output == {"value": "ok"}


def test_api_key_header_sent_only_when_env_var_is_set(server, monkeypatch):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"})))
    settings = _settings(server, api_key_env="DREAM_TEST_LOCAL_LLM_KEY")

    monkeypatch.delenv("DREAM_TEST_LOCAL_LLM_KEY", raising=False)
    openai_compatible.generate(_request(), settings)
    monkeypatch.setenv("DREAM_TEST_LOCAL_LLM_KEY", "sk-local-secret")
    result = openai_compatible.generate(_request(), settings)

    first, second = server.requests
    assert "authorization" not in first["headers"]
    assert second["headers"]["authorization"] == "Bearer sk-local-secret"
    assert "sk-local-secret" not in result.raw_result


def test_api_key_never_appears_in_errors(server, monkeypatch):
    monkeypatch.setenv("DREAM_TEST_LOCAL_LLM_KEY", "sk-local-secret")
    server.responder = lambda request: (401, {"error": {"message": "invalid api key"}})

    with pytest.raises(ProviderError) as excinfo:
        openai_compatible.generate(_request(), _settings(server, api_key_env="DREAM_TEST_LOCAL_LLM_KEY"))
    assert "sk-local-secret" not in str(excinfo.value)


def test_no_api_key_env_sends_no_authorization(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"})))

    openai_compatible.generate(_request(), _settings(server))

    assert "authorization" not in server.requests[0]["headers"]


def test_generate_refuses_non_loopback_url_without_egress_consent(server):
    with pytest.raises(ProviderError, match="remote_data_egress"):
        # A short timeout keeps a regression from waiting on a non-loopback connect.
        openai_compatible.generate(
            _request(timeout_seconds=1), {"type": "openai-compatible", "base_url": "http://192.0.2.10:8080"},
        )
    assert server.requests == []


def test_registry_builds_provider_with_settings(server):
    server.responder = lambda request: (200, _completion(json.dumps({"value": "ok"})))

    provider = create_provider("openai-compatible", _settings(server))

    assert provider.generate_structured(_request()).output == {"value": "ok"}


# Configuration


def _write_config(tmp_path: Path, provider_block: str, *, route: bool = True, stage_extra: str = "") -> Path:
    path = tmp_path / "config.toml"
    stages = f'[stages.distill]\nprovider = "local"\n{stage_extra}' if route else ""
    path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "dream.db"}"\nmemory_root = "{tmp_path / "memory"}"\n'
        f"{stages}[providers.local]\n{provider_block}",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("host", ["127.0.0.1", "127.8.9.10", "localhost", "LOCALHOST", "[::1]"])
def test_loopback_base_url_is_accepted_without_egress_consent(tmp_path, host):
    config = load_config(_write_config(tmp_path, f'type = "openai-compatible"\nbase_url = "http://{host}:8080"\n'))
    assert config.provider("local")["base_url"] == f"http://{host}:8080"


@pytest.mark.parametrize("url", [
    "http://192.168.1.20:8080",
    "https://api.example.com",
    "http://localhost.example.com:8080",
    "http://[::ffff:8.8.8.8]:8080",
    "http://0.0.0.0:8080",
])
def test_non_loopback_base_url_without_egress_consent_is_rejected(tmp_path, url):
    with pytest.raises(ConfigError, match="remote_data_egress"):
        load_config(_write_config(tmp_path, f'type = "openai-compatible"\nbase_url = "{url}"\n'))


def test_non_loopback_base_url_with_egress_consent_is_accepted(tmp_path):
    config = load_config(_write_config(
        tmp_path, 'type = "openai-compatible"\nbase_url = "https://llm.example.com"\nremote_data_egress = true\n',
    ))
    assert config.provider("local")["remote_data_egress"] is True


def test_unrouted_provider_is_still_validated(tmp_path):
    with pytest.raises(ConfigError, match="remote_data_egress"):
        load_config(_write_config(tmp_path, 'type = "openai-compatible"\nbase_url = "http://10.0.0.5"\n', route=False))


@pytest.mark.parametrize("block, message", [
    ('type = "openai-compatible"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = ""\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "ftp://127.0.0.1"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "127.0.0.1:8080"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\nremote_data_egress = "no"\n', "remote_data_egress"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\ntimeout_seconds = 0\n', "timeout_seconds"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\ntimeout_seconds = true\n', "timeout_seconds"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\ntimeout_seconds = 1.5\n', "timeout_seconds"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:0"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:99999"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\nmodel = ""\n', "model"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\napi_key_env = 5\n', "api_key_env"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\napi_key = "sk-inline"\n', "unknown"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1"\nexecutable = "llama"\n', "unknown"),
])
def test_invalid_openai_compatible_provider_is_rejected(tmp_path, block, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write_config(tmp_path, block))


def test_cli_provider_still_rejects_http_options(tmp_path):
    with pytest.raises(ConfigError, match="unknown provider local options"):
        load_config(_write_config(tmp_path, 'type = "codex-cli"\nbase_url = "http://127.0.0.1:8080"\n'))


# Preflight


def test_preflight_lists_models_without_calling_the_model(server):
    server.responder = lambda request: (200, {"object": "list", "data": [{"id": "qwen3-32b"}, {"id": "bge-m3"}]})

    ok, detail = preflight_provider(_settings(server))

    assert ok
    assert "qwen3-32b" in detail and "bge-m3" in detail
    assert [(request["method"], request["path"]) for request in server.requests] == [("GET", "/v1/models")]


def test_preflight_notes_configured_model_missing_from_listing(server):
    server.responder = lambda request: (200, {"object": "list", "data": [{"id": "other-model"}]})

    ok, detail = preflight_provider(_settings(server))

    assert ok
    assert "qwen3-32b" in detail and "not listed" in detail


def test_preflight_reports_unreachable_server_without_raising(server):
    url = server.base_url
    server.close()

    ok, detail = preflight_provider(_settings(server, base_url=url))

    assert not ok
    assert "unreachable" in detail


def test_preflight_reports_http_error_without_raising(server):
    server.responder = lambda request: (401, {"error": {"message": "invalid api key"}})

    ok, detail = preflight_provider(_settings(server))

    assert not ok
    assert "401" in detail


def test_backend_preflight_checks_unrouted_openai_compatible_providers(tmp_path, monkeypatch, server):
    monkeypatch.setenv("HOME", str(tmp_path))
    server.responder = lambda request: (200, {"object": "list", "data": [{"id": "qwen3-32b"}]})
    config = load_config(_write_config(
        tmp_path, f'type = "openai-compatible"\nbase_url = "{server.base_url}"\n', route=False,
    ))
    monkeypatch.setattr(backend, "preflight_provider", lambda cfg: (True, cfg["type"]))
    checks = backend.preflight(config)
    assert ("local", True, "openai-compatible") in checks
    assert ("codex", True, "codex-cli") in checks


def test_dream_preflight_command_reports_local_server(tmp_path, monkeypatch, server, capsys):
    import dream

    monkeypatch.setenv("HOME", str(tmp_path))
    server.responder = lambda request: (200, {"object": "list", "data": [{"id": "qwen3-32b"}]})
    path = _write_config(tmp_path, f'type = "openai-compatible"\nbase_url = "{server.base_url}"\nmodel = "qwen3-32b"\n')

    dream.main(["--config", str(path), "--db", str(tmp_path / "dream.db"), "preflight"])

    port = server.base_url.rsplit(":", 1)[1]
    assert f"ok  local: 127.0.0.1:{port} reachable; models: qwen3-32b" in capsys.readouterr().out.splitlines()
    assert [request["path"] for request in server.requests] == ["/v1/models"]
    assert not (tmp_path / "dream.db").exists()


def test_backend_preflight_scoped_to_stages_skips_unrouted_providers(tmp_path, monkeypatch, server):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(_write_config(
        tmp_path, f'type = "openai-compatible"\nbase_url = "{server.base_url}"\n', route=False,
    ))
    monkeypatch.setattr(backend, "preflight_provider", lambda cfg: (True, cfg["type"]))
    assert [name for name, *_ in backend.preflight(config, stages=["distill"])] == ["codex"]


# Distillation through the configured stage


def _distill_fixture(tmp_path: Path, monkeypatch, server, stage_extra: str = ""):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(_write_config(
        tmp_path, f'type = "openai-compatible"\nbase_url = "{server.base_url}"\nmodel = "qwen3-32b"\n',
        stage_extra=stage_extra,
    ))
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO sessions(session_id, project_slug, jsonl_path, jsonl_mtime, started_at, cwd, total_chars) "
        "VALUES ('s1', '-home-user-project', '/x.jsonl', 1.0, '2026-10-20T10:00:00Z', '/home/user/project', 2000)"
    )
    conn.execute("INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES ('s1', 0, 'user', NULL, 'Use the local model.')")
    conn.execute("INSERT INTO messages(session_id, seq, role, timestamp, text) VALUES ('s1', 1, 'assistant', NULL, 'Done.')")
    conn.commit()
    return config, conn


def test_distill_records_openai_compatible_provenance(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(
        json.dumps(DISTILL_OUTPUT), model="qwen3-32b-q4_k_m.gguf",
        usage={"prompt_tokens": 900, "completion_tokens": 80},
    ))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)

    result = distill_session(conn, "s1", config=config)

    assert result is not None and result.provider == "openai-compatible"
    row = conn.execute("SELECT provider, model, input_tokens, output_tokens, usage_json FROM distilled").fetchone()
    assert row[:4] == ("openai-compatible", "qwen3-32b-q4_k_m.gguf", 900, 80)
    assert json.loads(row[4])["input_tokens"] == 900
    [request] = server.chat_requests()
    assert request["body"]["response_format"]["json_schema"]["schema"] == DISTILL_SCHEMA
    assert request["body"]["messages"][0]["content"] == backend.SYSTEM_PROMPT


def test_distill_timeout_is_provider_failure_and_writes_no_row(tmp_path, monkeypatch, server):
    def responder(request):
        server.release.wait(5)
        return 200, _completion(json.dumps(DISTILL_OUTPUT))

    server.responder = responder
    # The stage timeout wins over the provider's default of 600 s.
    config, conn = _distill_fixture(tmp_path, monkeypatch, server, stage_extra="timeout_seconds = 1\n")

    with pytest.raises(ProviderError, match="timed out"):
        distill_session(conn, "s1", config=config)
    assert conn.execute("SELECT COUNT(*) FROM distilled").fetchone()[0] == 0
    assert len(server.chat_requests()) == 1
