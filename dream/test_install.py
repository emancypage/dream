import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_installer_links_the_skill_for_codex_and_claude(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    subprocess.run(["bash", str(REPO / "install.sh")], check=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
                   capture_output=True)
    for link in (home / ".codex/skills/dream", home / ".claude/skills/dream"):
        assert link.is_symlink()
        assert link.resolve() == REPO / "skill"
