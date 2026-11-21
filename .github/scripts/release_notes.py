import json
import re
from pathlib import Path


def release_notes(changelog: str, version: str) -> str:
    heading = r"^#{2,3} \[?v?(\d+\.\d+\.\d+)(?=\]|\s|$)[^\n]*$"
    sections = re.split(heading, changelog, flags=re.MULTILINE)
    if len(sections) < 3 or sections[1] != version:
        raise ValueError(f"Expected the newest changelog entry to be {version}")
    notes = sections[2].strip()
    if not notes:
        raise ValueError(f"Empty changelog entry for {version}")
    return notes


if __name__ == "__main__":
    version = Path("VERSION").read_text().strip()
    manifest = json.loads(Path(".release-please-manifest.json").read_text())
    if version != manifest["."]:
        raise SystemExit("VERSION and the release manifest disagree")
    print(release_notes(Path("CHANGELOG.md").read_text(), version))
