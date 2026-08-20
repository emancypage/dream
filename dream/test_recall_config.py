"""Task 1: recall contracts and configuration.

Covers default values, strict rejection of unknown/invalid core recall keys,
and non-blocking recording of invalid optional adapter configuration.
"""

import dataclasses
from pathlib import Path

import pytest

from config import ConfigError, load_config
from recall_types import (
    AdapterSettings,
    CalibrationRecord,
    RecallCandidate,
    RecallDiagnostics,
    RecallDocument,
    RecallQuery,
    RecallResult,
    RecallSettings,
)


DEFAULT_ADAPTER = AdapterSettings(enabled=False, type="none", remote_data_egress=False)


def _write(tmp_path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_recall_defaults_match_packaged_configuration():
    config = load_config(Path("/nonexistent/dream-recall-config.toml"))
    settings = config.recall_settings()
    assert isinstance(settings, RecallSettings)
    assert settings.enabled is True
    assert settings.install_hooks is False
    assert settings.allow_raw_transcript_prompt is False
    assert settings.first_prompt_only is True
    assert settings.session_start_budget_codepoints == 6000
    assert settings.prompt_budget_codepoints == 4000
    assert settings.session_start_additional_context_limit == 1800
    assert settings.prompt_additional_context_limit == 1200
    assert settings.diagnostic_path == "~/.cache/dream/recall-diagnostics.jsonl"
    assert settings.calibration_path == "~/.config/dream/recall-calibration.json"
    assert settings.embedder == DEFAULT_ADAPTER
    assert settings.reranker == DEFAULT_ADAPTER
    assert config.optional_config_errors == ()


def test_recall_settings_are_immutable():
    config = load_config(Path("/nonexistent/dream-recall-config.toml"))
    settings = config.recall_settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.enabled = False


def test_unknown_recall_core_key_is_rejected(tmp_path):
    config_path = _write(tmp_path, "[recall]\nunknown_key = true\n")
    with pytest.raises(ConfigError, match="unknown recall options"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('[recall]\nenabled = "yes"\n', "recall.enabled must be a boolean"),
        ("[recall]\nsession_start_budget_codepoints = \"6000\"\n", "recall.session_start_budget_codepoints must be a positive integer"),
        ("[recall]\nsession_start_budget_codepoints = -1\n", "recall.session_start_budget_codepoints must be a positive integer"),
        ('[recall]\ndiagnostic_path = ""\n', "recall.diagnostic_path must be a non-empty string"),
        ("[recall]\ninstall_hooks = 1\n", "recall.install_hooks must be a boolean"),
    ],
)
def test_invalid_recall_core_value_is_rejected(tmp_path, body, expected):
    config_path = _write(tmp_path, body)
    with pytest.raises(ConfigError, match=expected):
        load_config(config_path)


def test_invalid_optional_embedder_config_is_recorded_without_blocking(tmp_path):
    config_path = _write(tmp_path, "[recall.embedder]\nenabled = \"yes\"\n")
    config = load_config(config_path)
    assert any("recall.embedder.enabled must be a boolean" in error for error in config.optional_config_errors)
    # Core configuration still loads and stays intact.
    settings = config.recall_settings()
    assert settings.enabled is True
    assert settings.session_start_budget_codepoints == 6000
    # The invalid adapter falls back to its safe disabled default.
    assert settings.embedder == DEFAULT_ADAPTER


def test_unknown_optional_adapter_key_is_recorded(tmp_path):
    config_path = _write(tmp_path, "[recall.reranker]\nmodel = \"gpt-x\"\n")
    config = load_config(config_path)
    assert any("unknown recall.reranker options: model" in error for error in config.optional_config_errors)
    assert config.recall_settings().reranker == DEFAULT_ADAPTER


def test_non_table_adapter_value_is_recorded_without_blocking(tmp_path):
    config_path = _write(tmp_path, '[recall]\nembedder = "none"\n')
    config = load_config(config_path)
    assert any("recall.embedder" in error for error in config.optional_config_errors)
    assert config.recall_settings().embedder == DEFAULT_ADAPTER


def test_valid_optional_adapter_config_is_applied(tmp_path):
    config_path = _write(
        tmp_path,
        "[recall]\n"
        "first_prompt_only = false\n"
        "\n"
        "[recall.embedder]\n"
        "enabled = true\n"
        'type = "none"\n'
        "remote_data_egress = true\n",
    )
    config = load_config(config_path)
    assert config.optional_config_errors == ()
    settings = config.recall_settings()
    assert settings.first_prompt_only is False
    assert settings.embedder == AdapterSettings(enabled=True, type="none", remote_data_egress=True)
    assert settings.reranker == DEFAULT_ADAPTER


def test_recall_contract_types_are_fully_specified_and_immutable():
    document = RecallDocument(
        id="0d9b",
        content_sha256="sha256-value",
        source_kind="approved_memory",
        trust_level="user_approved",
        project_slug=None,
        source_path="api.md",
        source_updated_at="2026-08-20T10:00:00Z",
        indexed_at="2026-08-20T10:00:01Z",
        source_version="v1",
        text="canonical searchable text",
    )
    query = RecallQuery(
        query_text="postgres migration",
        session_id=None,
        hook_event="prompt",
        cwd=None,
        repository_roots=("/home/a/api",),
        requested_codepoint_budget=4000,
        excluded_source_ids=frozenset({"0d9b"}),
        allow_raw_transcript=False,
    )
    candidate = RecallCandidate(
        document=document,
        scrubbed_excerpt="excerpt",
        component_ranks={"fts": 1, "recency": 4},
        score=0.016,
        best_component_rank=1,
    )
    diagnostics = RecallDiagnostics(
        candidate_count=3,
        selected_count=1,
        fallback_reason=None,
        selected_codepoints=10,
        elapsed_ms=5,
        calibration_version=None,
    )
    result = RecallResult(
        selected=(candidate,),
        rendered_context="[Dream recall]",
        mode="lexical",
        calibration_version=None,
        diagnostics=diagnostics,
    )
    calibration = CalibrationRecord(
        mode="lexical",
        calibration_version="cv-1",
        threshold=0.1,
        fixture_sha256="fixture-hash",
        created_at="2026-08-20T10:00:00Z",
    )
    for instance, field_name in (
        (document, "id"),
        (query, "query_text"),
        (candidate, "score"),
        (result, "mode"),
        (diagnostics, "elapsed_ms"),
        (calibration, "threshold"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, "mutated")
