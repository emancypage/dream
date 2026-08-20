from pathlib import Path

from recall_scrub import scrub_text


def test_scrub_removes_common_secrets_and_home_paths(tmp_path):
    home = tmp_path / "home" / "user"
    value = (
        f"path={home}/project PEM -----BEGIN PRIVATE KEY-----abc-----END PRIVATE KEY----- "
        "Bearer abc.def.ghi password=secret postgres://user:pass@example/db "
        "OPENAI_API_KEY=sk-test"
    )
    scrubbed = scrub_text(value, home)
    assert "<home>" in scrubbed
    assert "<redacted>" in scrubbed
    assert "sk-test" not in scrubbed
    assert "user:pass" not in scrubbed
