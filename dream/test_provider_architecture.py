import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import ConfigError, load_config
from model_types import SchemaValidationError, validate_output


ROOT = Path(__file__).parent


def test_default_profile_routes_both_stages_to_codex():
    config = load_config(Path("/nonexistent/dream-config.toml"))
    assert config.stage("distill")["provider"] == "codex"
    assert config.stage("consolidate")["provider"] == "codex"
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
