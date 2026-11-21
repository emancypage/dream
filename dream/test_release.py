import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
NOTES_SCRIPT = REPO / ".github" / "scripts" / "release_notes.py"


def test_version_from_installed_symlink(tmp_path):
    launcher = tmp_path / "dream"
    launcher.symlink_to(REPO / "dream" / "dream.py")
    output = subprocess.check_output([launcher, "--version"], cwd=tmp_path, text=True)
    assert output.strip() == f"dream {(REPO / 'VERSION').read_text().strip()}"


@pytest.mark.parametrize("header", [
    "## 0.1.0 (2026-11-21)",
    "## [0.1.0](compare/v0.0.0...v0.1.0) (2026-11-21)",
])
def test_release_uses_reviewed_notes_for_one_version(tmp_path, header):
    notes = "### Features\n\n* Search sessions.\n\n### Upgrade\n\nRun `dream preflight`."
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n{header}\n\n{notes}\n\n### 0.0.9\n\n* Older fix.\n"
    )
    (tmp_path / "VERSION").write_text("0.1.0\n")
    (tmp_path / ".release-please-manifest.json").write_text(json.dumps({".": "0.1.0"}))
    output = subprocess.check_output([sys.executable, NOTES_SCRIPT], cwd=tmp_path, text=True)
    assert output == notes + "\n"


@pytest.mark.parametrize(("manifest_version", "entry", "error"), [
    ("0.2.0", "## 0.1.0\n\n* Fix.", "manifest disagree"),
    ("0.1.0", "## 0.1.1\n\n* Fix.", "Expected the newest"),
    ("0.1.0", "## 0.1.0\n\n## 0.0.9\n\n* Fix.", "Empty changelog entry"),
])
def test_inconsistent_release_fails_before_publishing(tmp_path, manifest_version, entry, error):
    (tmp_path / "VERSION").write_text("0.1.0\n")
    (tmp_path / ".release-please-manifest.json").write_text(json.dumps({".": manifest_version}))
    (tmp_path / "CHANGELOG.md").write_text(entry)
    result = subprocess.run([sys.executable, NOTES_SCRIPT], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert error in result.stderr
