import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import ConfigError, load_config
from model_types import SchemaValidationError, validate_output


ROOT = Path(__file__).parent


def test_nightly_service_invokes_configured_curation_not_mechanical_apply():
    service = (ROOT / "systemd" / "dream.service").read_text(encoding="utf-8")
    assert "dream suggestions curate-configured" in service
    assert "dream suggestions apply-configured" not in service
    assert "claude" not in service.lower()


def test_nightly_service_continues_after_individual_stage_failures():
    service = (ROOT / "systemd" / "dream.service").read_text(encoding="utf-8")
    stages = [line for line in service.splitlines() if line.startswith("ExecStart=")]
    assert len(stages) == 6
    assert all("dream-stage --continue-on-error " in line for line in stages)


def test_default_profile_routes_both_stages_to_codex():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    assert config.stage("distill")["provider"] == "codex"
    assert config.stage("consolidate")["provider"] == "codex"
    assert config.stage("review") == {
        "provider": "codex",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "high",
        "timeout_seconds": 1800,
    }
    assert config.provider("codex")["type"] == "codex-cli"


def test_unknown_configuration_option_is_rejected(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[review]\nmode='suggest-only'\nunknown=true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown review options"):
        load_config(config_path)


def test_schema_validation_rejects_invalid_provider_output():
    schema = {
        "type": "object",
        "required": ["value"],
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
    }
    with pytest.raises(SchemaValidationError):
        validate_output({"value": 1}, schema)


def test_codex_adapter_import_does_not_import_claude_adapter():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from providers.registry import create_provider; "
        "create_provider('codex-cli'); "
        "assert 'claude_cli' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_openai_compatible_adapter_import_does_not_import_cli_adapters():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from providers.registry import create_provider; "
        "create_provider('openai-compatible', {'type': 'openai-compatible', 'base_url': 'http://127.0.0.1:8080'}); "
        "assert 'codex_cli' not in sys.modules; "
        "assert 'claude_cli' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


def test_unsupported_provider_type_is_rejected(tmp_path):
    from model_types import ProviderError
    from providers.registry import create_provider

    with pytest.raises(ProviderError, match="unsupported provider type"):
        create_provider("http-json")
    config_path = tmp_path / "config.toml"
    config_path.write_text("[providers.local]\ntype = 'http-json'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported type"):
        load_config(config_path)


def test_default_profile_defines_no_openai_compatible_provider():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    assert {name: provider["type"] for name, provider in config.data["providers"].items()} == {
        "codex": "codex-cli",
        "claude": "claude-cli",
    }


def test_default_config_example_openai_compatible_provider_is_valid(tmp_path):
    lines = (ROOT / "default-config.toml").read_text(encoding="utf-8").splitlines()
    start = lines.index("# [providers.local-qwen]")
    example = []
    for line in lines[start:]:
        if not line.startswith("# "):
            break
        example.append(line[2:])
    config_path = tmp_path / "config.toml"
    config_path.write_text("\n".join(example) + "\n", encoding="utf-8")
    provider = load_config(config_path).provider("local-qwen")
    assert provider["type"] == "openai-compatible"
    assert provider["remote_data_egress"] is False
    assert provider["timeout_seconds"] == 600


def test_default_profile_enables_opencode_sqlite_and_keeps_claude_disabled():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    sources = {source["name"]: source for source in config.sources}
    assert sources["opencode"]["type"] == "opencode-sqlite"
    assert sources["opencode"]["enabled"] is True
    assert sources["opencode"]["root"] == "~/.local/share/opencode"
    assert sources["claude-history"]["enabled"] is False


def test_module_help_describes_auto_apply_memory_writes():
    import dream

    assert "auto-apply" in dream.__doc__
    assert "never to live memory files" not in dream.__doc__
    assert "writes suggestions only" in dream.__doc__
