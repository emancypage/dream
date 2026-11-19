import pytest

from config import ConfigError, DelegateSettings, load_config


def test_delegate_is_off_by_default(tmp_path):
    assert load_config(tmp_path / "absent").delegate_settings() == DelegateSettings()


@pytest.mark.parametrize("value", [
    {"enabled": "yes"},
    {"compact_max_files_per_run": 0},
    {"compact_target_chars": 40_000},
    {"compact_over_chars": 60_000, "compact_target_chars": 25_000},
    {"unknown": 1},
])
def test_invalid_delegate_values_are_rejected(tmp_path, value):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent", {"delegate": value})


def test_enabled_delegate_requires_an_explicit_model(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "absent", {"delegate": {"enabled": True}})
    cfg = load_config(tmp_path / "absent", {"delegate": {"enabled": True},
                                            "stages": {"delegate": {"model": "fake-model"}}})
    assert cfg.delegate_settings().enabled is True


@pytest.mark.parametrize("enabled", [False, True])
def test_default_preflight_includes_delegate_only_when_enabled(tmp_path, monkeypatch, enabled):
    import backend
    stages = {stage: {"provider": "claude"} for stage in ("distill", "consolidate", "audit", "review")}
    stages["delegate"] = {"provider": "codex", "model": "fake"}
    cfg = load_config(tmp_path / "absent", {"stages": stages, "delegate": {"enabled": enabled}})
    monkeypatch.setattr(backend, "preflight_provider", lambda cfg: (cfg["type"] != "codex-cli", cfg["type"]))
    results = backend.preflight(cfg)
    assert [name for name, *_ in results] == (["claude", "codex"] if enabled else ["claude"])
    assert backend.preflight(cfg, stages=["delegate"])[0][:2] == ("codex", False)
