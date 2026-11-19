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
