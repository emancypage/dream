import json

import pytest

from recall_hooks import MARKER, install_hooks, uninstall_hooks, validate_hooks_document


def test_install_is_repeatable_and_preserves_unrelated_hooks(tmp_path):
    path = tmp_path / "hooks.json"
    path.write_text(json.dumps({"hooks": {"Other": [{"hooks": [{"type": "command", "command": "keep"}]}]}}), encoding="utf-8")
    first = install_hooks(path)
    second = install_hooks(path)
    document = json.loads(path.read_text())
    assert first.installed == second.installed == 2
    assert document["hooks"]["Other"][0]["hooks"][0]["command"] == "keep"
    assert sum(len(group["hooks"]) for group in document["hooks"]["SessionStart"]) == 1
    assert second.backup_path


def test_install_preserves_unrelated_group_in_same_event(tmp_path):
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps({
            "hooks": {
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "keep"}]}
                ]
            }
        }),
        encoding="utf-8",
    )
    install_hooks(path)
    document = json.loads(path.read_text())
    commands = [
        command["command"]
        for group in document["hooks"]["UserPromptSubmit"]
        for command in group["hooks"]
    ]
    assert commands == ["keep", "dream context prompt"]


def test_uninstall_is_selective_and_repeatable(tmp_path):
    path = tmp_path / "hooks.json"
    install_hooks(path)
    document = json.loads(path.read_text())
    document["hooks"]["SessionStart"].append({"matcher": "other", "hooks": [{"type": "command", "command": "keep"}]})
    path.write_text(json.dumps(document), encoding="utf-8")
    report = uninstall_hooks(path)
    assert report.removed == 2
    after = json.loads(path.read_text())
    assert after["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "keep"
    assert not uninstall_hooks(path).changed


def test_malformed_input_does_not_mutate(tmp_path):
    path = tmp_path / "hooks.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        install_hooks(path)
    assert path.read_text() == "{broken"


def test_generated_commands_are_relative_and_document_valid(tmp_path):
    path = tmp_path / "hooks.json"
    install_hooks(path, "/usr/local/bin/dream")
    document = json.loads(path.read_text())
    validate_hooks_document(document)
    commands = [command["command"] for groups in document["hooks"].values() for group in groups for command in group["hooks"]]
    assert all(not command.startswith("/") for command in commands)
    assert all(command.get("statusMessage") == MARKER for groups in document["hooks"].values() for group in groups for command in group["hooks"] if "statusMessage" in command)
