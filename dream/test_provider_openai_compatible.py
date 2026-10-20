"""OpenAI-compatible generation provider against a loopback fake server."""

import json
import sqlite3
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import backend
from config import ConfigError, load_config
from distill import DISTILL_SCHEMA, distill_session, prompt_version, sessions_needing_distill
from model_types import GenerationRequest, GenerationResult, ProviderError, SchemaValidationError
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


def test_server_error_naming_response_format_is_not_retried(server):
    server.responder = lambda request: (500, {"error": {"message": "Unsupported response_format"}})

    with pytest.raises(ProviderError, match="500"):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 1


def test_bad_request_unrelated_to_response_format_is_not_retried(server):
    server.responder = lambda request: (400, {"error": {"message": "the request exceeds the available context size"}})

    with pytest.raises(ProviderError, match="400"):
        openai_compatible.generate(_request(), _settings(server))
    assert len(server.chat_requests()) == 1


def test_fallback_output_is_validated_host_side(server):
    def responder(request):
        if request["body"]["response_format"]["type"] == "json_schema":
            return 422, {"error": "json_schema is not supported by this server"}
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


def test_connect_timeout_is_a_provider_error(server, monkeypatch):
    class TimingOutOpener:
        def open(self, request, timeout):
            raise urllib.error.URLError(TimeoutError("connect timed out"))

    monkeypatch.setattr(openai_compatible, "_opener", lambda: TimingOutOpener())

    with pytest.raises(ProviderError, match="timed out after 30s"):
        openai_compatible.generate(_request(), _settings(server))


def test_oversized_response_body_is_rejected(server, monkeypatch):
    monkeypatch.setattr(openai_compatible, "_RESPONSE_BODY_LIMIT", 64)
    server.responder = lambda request: (200, _completion(json.dumps({"value": "x" * 200})))

    with pytest.raises(ProviderError, match="exceeds 64 bytes"):
        openai_compatible.generate(_request(), _settings(server))


def test_response_body_at_the_limit_is_accepted(server, monkeypatch):
    body = json.dumps(_completion(json.dumps({"value": "ok"}))).encode("utf-8")
    monkeypatch.setattr(openai_compatible, "_RESPONSE_BODY_LIMIT", len(body))
    server.responder = lambda request: (200, body)

    assert openai_compatible.generate(_request(), _settings(server)).output == {"value": "ok"}


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


def test_api_key_echoed_by_server_is_redacted_from_errors(server, monkeypatch):
    monkeypatch.setenv("DREAM_TEST_LOCAL_LLM_KEY", "sk-local-secret")
    server.responder = lambda request: (
        403, {"error": {"message": "rejected header " + request["headers"].get("authorization", "")}},
    )

    with pytest.raises(ProviderError) as excinfo:
        openai_compatible.generate(_request(), _settings(server, api_key_env="DREAM_TEST_LOCAL_LLM_KEY"))
    assert "sk-local-secret" not in str(excinfo.value)
    assert "Bearer [redacted]" in str(excinfo.value)
    ok, detail = preflight_provider(_settings(server, api_key_env="DREAM_TEST_LOCAL_LLM_KEY"))
    assert not ok and "sk-local-secret" not in detail


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


@pytest.mark.parametrize("suffix", ["/#@evil.com", "/?@evil.com", "?", "#"])
def test_generate_refuses_base_url_with_query_or_fragment(server, suffix):
    with pytest.raises(ProviderError, match="base_url"):
        openai_compatible.generate(_request(timeout_seconds=1), _settings(server, base_url=server.base_url + suffix))
    assert server.requests == []


def test_generate_refuses_base_url_with_credentials(server):
    url = server.base_url.replace("http://", "http://evil.com@")
    with pytest.raises(ProviderError, match="base_url"):
        openai_compatible.generate(_request(timeout_seconds=1), _settings(server, base_url=url))
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
    ('type = "openai-compatible"\nbase_url = "http://evil.com@127.0.0.1:8080"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://user:pass@127.0.0.1:8080"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:8080/#@evil.com"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:8080/?@evil.com"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:8080?"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:8080/?"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:8080#"\n', "base_url"),
    ('type = "openai-compatible"\nbase_url = "http://127.0.0.1:8080/v1#"\n', "base_url"),
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


def test_backend_preflight_skips_unused_openai_compatible_providers(tmp_path, monkeypatch, server):
    # Same rule as an unused [providers.claude]: only routed providers can fail preflight.
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(_write_config(
        tmp_path, f'type = "openai-compatible"\nbase_url = "{server.base_url}"\n', route=False,
    ))
    monkeypatch.setattr(backend, "preflight_provider", lambda cfg: (True, cfg["type"]))
    assert [name for name, *_ in backend.preflight(config)] == ["codex"]
    assert server.requests == []


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


# Refresh compares recorded stage settings, not the server-reported model


def _pending(conn, config, *, refresh=True) -> list[str]:
    return [sid for sid, _ in sessions_needing_distill(conn, config=config, refresh_config=refresh)]


def _routed_config(
    tmp_path: Path, server, *, provider: str = "local", stage_extra: str = "",
    provider_model: str = "qwen3-32b", base_url: str | None = None,
):
    path = tmp_path / f"config-{provider}.toml"
    url = base_url or server.base_url
    path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "dream.db"}"\nmemory_root = "{tmp_path / "memory"}"\n'
        f'[stages.distill]\nprovider = "{provider}"\n{stage_extra}'
        f'[providers.local]\ntype = "openai-compatible"\nbase_url = "{url}"\nmodel = "{provider_model}"\n'
        f'[providers.local2]\ntype = "openai-compatible"\nbase_url = "{url}"\nmodel = "{provider_model}"\n',
        encoding="utf-8",
    )
    return load_config(path)


def test_refresh_requeues_when_provider_level_model_changes(tmp_path, monkeypatch, server):
    # The packaged example puts model on the provider table and leaves the stage without one.
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT), model="qwen3-32b-q4_k_m.gguf"))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config)

    assert _pending(conn, _routed_config(tmp_path, server)) == []
    assert _pending(conn, _routed_config(tmp_path, server, provider_model="qwen3-14b")) == ["s1"]
    assert _pending(conn, _routed_config(tmp_path, server, provider_model="qwen3-14b"), refresh=False) == []


def test_refresh_requeues_when_base_url_changes(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT)))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config)

    assert _pending(conn, _routed_config(tmp_path, server, base_url="http://127.0.0.1:18081")) == ["s1"]


def test_refresh_compares_the_effective_model_not_where_it_is_set(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT)))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config)

    # Naming the same model on the stage leaves the request unchanged.
    assert _pending(conn, _routed_config(tmp_path, server, stage_extra='model = "qwen3-32b"\n')) == []
    # A stage model overrides a changed provider model, so the request is unchanged too.
    assert _pending(conn, _routed_config(
        tmp_path, server, stage_extra='model = "qwen3-32b"\n', provider_model="qwen3-14b",
    )) == []


def test_refresh_honours_the_model_override(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT)))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config, model="qwen3-14b")
    routed = _routed_config(tmp_path, server)

    def pending(model):
        return [sid for sid, _ in sessions_needing_distill(conn, config=routed, refresh_config=True, model=model)]

    # Rerunning with the same override finds nothing new; dropping it compares against the configured model.
    assert pending("qwen3-14b") == []
    assert pending(None) == ["s1"]


def test_recorded_route_snapshot_is_credential_free(tmp_path, monkeypatch, server):
    monkeypatch.setenv("DREAM_TEST_LOCAL_LLM_KEY", "sk-local-secret")
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT)))
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_config(_write_config(
        tmp_path,
        f'type = "openai-compatible"\nbase_url = "{server.base_url}"\nmodel = "qwen3-32b"\n'
        'api_key_env = "DREAM_TEST_LOCAL_LLM_KEY"\n',
    ))
    _config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config)

    assert server.chat_requests()[0]["headers"]["authorization"] == "Bearer sk-local-secret"
    options_text = conn.execute("SELECT provider_options FROM distilled").fetchone()[0]
    assert "sk-local-secret" not in options_text and "DREAM_TEST_LOCAL_LLM_KEY" not in options_text
    options = json.loads(options_text)
    assert options["effective_request"] == {
        "provider": "local", "type": "openai-compatible", "base_url": server.base_url, "model": "qwen3-32b",
    }
    assert {key: value for key, value in options.items() if key != "effective_request"} == config.stage("distill")


def test_cli_route_provider_options_are_byte_identical(tmp_path, monkeypatch, server):
    import distill

    _config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    config = _codex_config(tmp_path)
    monkeypatch.setattr(distill, "generate", lambda *args, **kwargs: GenerationResult(
        output=DISTILL_OUTPUT, raw_result="{}", provider="codex", model=None, usage=None, duration_ms=1,
    ))
    distill_session(conn, "s1", config=config)

    stored = conn.execute("SELECT provider_options FROM distilled").fetchone()[0]
    assert stored == json.dumps(config.stage("distill"), sort_keys=True)


def _legacy_row(conn, config, *, served_model: str) -> None:
    """A row written before the effective-request snapshot existed."""
    conn.execute(
        "INSERT INTO distilled(session_id, provider, model, prompt_version, provider_options, notes_json) "
        "VALUES ('s1', 'openai-compatible', ?, ?, ?, '{}')",
        (served_model, prompt_version(), json.dumps(config.stage("distill"), sort_keys=True)),
    )


def test_legacy_row_requeues_when_provider_model_cannot_be_confirmed(tmp_path, monkeypatch, server):
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    _legacy_row(conn, config, served_model="qwen3-32b")

    # The server reported the requested name, so the unchanged route is confirmed.
    assert _pending(conn, config) == []
    # The probe case: provider-level model old -> new with an unchanged stage.
    assert _pending(conn, _routed_config(tmp_path, server, provider_model="qwen3-14b")) == ["s1"]
    # A server-specific name cannot confirm which model was requested: re-run it.
    conn.execute("UPDATE distilled SET model = 'qwen3-32b-q4_k_m.gguf'")
    assert _pending(conn, config) == ["s1"]


def test_legacy_row_with_stage_model_is_confirmed_by_the_stage(tmp_path, monkeypatch, server):
    config, conn = _distill_fixture(tmp_path, monkeypatch, server, stage_extra='model = "qwen3-14b"\n')
    _legacy_row(conn, config, served_model="served-name")

    assert _pending(conn, config) == []
    assert _pending(conn, _routed_config(tmp_path, server, stage_extra='model = "qwen3-8b"\n')) == ["s1"]


def test_refresh_keeps_openai_compatible_session_distilled_until_route_changes(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT), model="qwen3-32b-q4_k_m.gguf"))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config)

    assert _pending(conn, config) == []
    assert _pending(conn, config, refresh=False) == []
    assert _pending(conn, _routed_config(tmp_path, server)) == []
    assert _pending(conn, _routed_config(tmp_path, server, stage_extra='model = "qwen3-14b"\n')) == ["s1"]
    assert _pending(conn, _routed_config(tmp_path, server, provider="local2")) == ["s1"]
    assert _pending(conn, _routed_config(tmp_path, server, provider="codex")) == ["s1"]


def test_refresh_keeps_stage_model_route_distilled(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT), model="served-name"))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server, stage_extra='model = "qwen3-14b"\n')
    distill_session(conn, "s1", config=config)

    assert _pending(conn, config) == []
    assert _pending(conn, _routed_config(tmp_path, server)) == ["s1"]


def test_refresh_without_recorded_stage_settings_keeps_column_comparison(tmp_path, monkeypatch, server):
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    conn.execute(
        "INSERT INTO distilled(session_id, provider, model, prompt_version, provider_options, notes_json) "
        "VALUES ('s1', 'openai-compatible', 'qwen3-32b-q4_k_m.gguf', ?, NULL, '{}')",
        (prompt_version(),),
    )

    assert _pending(conn, config) == ["s1"]
    assert _pending(conn, config, refresh=False) == []


def test_refresh_requeues_row_recorded_by_another_provider_type_under_same_name(tmp_path, monkeypatch, server):
    # [providers.local] used to be codex-cli; the stage settings are identical.
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    conn.execute(
        "INSERT INTO distilled(session_id, provider, model, prompt_version, provider_options, notes_json) "
        "VALUES ('s1', 'codex', NULL, ?, ?, '{}')",
        (prompt_version(), json.dumps(config.stage("distill"), sort_keys=True)),
    )

    assert _pending(conn, config) == ["s1"]
    assert _pending(conn, config, refresh=False) == []


def test_refresh_still_detects_content_changes_on_openai_compatible_route(tmp_path, monkeypatch, server):
    server.responder = lambda request: (200, _completion(json.dumps(DISTILL_OUTPUT)))
    config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    distill_session(conn, "s1", config=config)
    conn.execute("UPDATE distilled SET prompt_version = 'older-prompt'")

    assert _pending(conn, config) == ["s1"]


def _codex_config(tmp_path: Path, stage_extra: str = ""):
    path = tmp_path / "codex.toml"
    path.write_text(
        f'[storage]\ndb_path = "{tmp_path / "dream.db"}"\nmemory_root = "{tmp_path / "memory"}"\n'
        f'[stages.distill]\nprovider = "codex"\n{stage_extra}',
        encoding="utf-8",
    )
    return load_config(path)


def test_codex_refresh_comparison_is_unchanged(tmp_path, monkeypatch, server):
    monkeypatch.setenv("HOME", str(tmp_path))
    _config, conn = _distill_fixture(tmp_path, monkeypatch, server)
    stage_options = json.dumps({"provider": "codex", "reasoning_effort": "low", "timeout_seconds": 600}, sort_keys=True)
    conn.execute(
        "INSERT INTO distilled(session_id, provider, model, prompt_version, provider_options, notes_json) "
        "VALUES ('s1', 'codex', NULL, ?, ?, '{}')",
        (prompt_version(), stage_options),
    )

    assert _pending(conn, _codex_config(tmp_path)) == []
    assert _pending(conn, _codex_config(tmp_path, 'model = "m-1"\n')) == ["s1"]
    # Recorded stage settings never override the column comparison for CLI routes.
    conn.execute("UPDATE distilled SET provider = 'openai-compatible'")
    assert _pending(conn, _codex_config(tmp_path)) == ["s1"]
