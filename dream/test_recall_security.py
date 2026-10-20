from pathlib import Path

import pytest

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


# Bare provider tokens. Values are assembled from fragments so that no
# token-shaped literal sits in the repository.
_MIXED = "A1b2C3d4E5f6G7h8J9k0"


def _join(*parts):
    return "".join(parts)


BARE_TOKENS = {
    "openai legacy": _join("sk", "-", _MIXED, _MIXED, "Zz9Yx8W7"),
    "openai lowercase": _join("sk", "-", "abcdefghijklmnopqrstuvwx"),
    "openai project": _join("sk", "-proj-", _MIXED, "_", _MIXED, "-", _MIXED),
    "openai service account": _join("sk", "-svcacct-", _MIXED, "-", _MIXED),
    "openai user key": _join("sk", "-None-", _MIXED, "_", _MIXED, "Zz9Y"),
    "anthropic": _join("sk", "-ant-", "api03-", "Ab-", _MIXED, _MIXED, _MIXED),
    "github classic": _join("gh", "p_", _MIXED, "K1m2N3o4P5q6R7s8"),
    "github oauth": _join("gh", "o_", _MIXED, "K1m2N3o4P5q6R7s8"),
    "github app": _join("gh", "s_", _MIXED, "K1m2N3o4P5q6R7s8"),
    "github fine-grained": _join("github", "_pat_", "11ABCDEFG0", _MIXED, "_", _MIXED, _MIXED),
    "aws access key": _join("AK", "IA", "Z7Q2", "M4XK", "P9RT", "B3WD"),
    "aws session key": _join("AS", "IA", "Y34F", "ZKBO", "KMUT", "VV7A"),
    "slack bot": _join("xo", "xb-", "123456789012-", "1234567890123-", "AbCdEfGhIjKlMnOpQrStUvWx"),
    "slack user": _join("xo", "xp-", "123456789012-", "1234567890123-", "1234567890123-", "0123456789abcdef"),
    "slack app": _join("xo", "xa-", "2-", "123456789012-", "AbCdEfGhIjKl"),
    "slack app-level": _join("xa", "pp-", "1-", "A0123456789-", "1234567890123-", "0123456789abcdef" * 4),
}


@pytest.mark.parametrize("name", sorted(BARE_TOKENS))
@pytest.mark.parametrize("template", ["{}", "token {} in a prompt", "use ({}).", 'key "{}", then', "a\n{}\nb"])
def test_bare_provider_tokens_are_redacted(tmp_path, name, template):
    token = BARE_TOKENS[name]
    scrubbed = scrub_text(template.format(token), tmp_path / "home")
    assert scrubbed == template.format("<redacted>")


@pytest.mark.parametrize("text", [
    "pip install scikit-learn; import sk-learn style APIs",
    "the sk-learn-compatible-estimator-wrapper-module pattern",
    _join("ta", "sk-", "abcdefghijklmnopqrstuvwxyz and ", "de", "sk-", "1234567890", "abcdefghij"),
    "the sk-admin-dashboard-settings-panel component",
    "CSS: .sk-proj-management-dashboard-v2-final { color: red }",
    "Branch feature/sk-proj-onboarding-checklist-redesign merged.",
    _join("sk", "-proj-", "lowercase-and-digits-only-0123456789-abcdefghij"),
    _join("sk", "-None-", "NoDigitsInThisLongMixedCaseKebabStyleNameHere"),
    _join("xa", "pp-", "1-", "short"), "xapp-store-review-notes-for-the-release",
    _join("sk", "-proj-", "short"),
    _join("sk", "-ant-", "api03-", "short"),
    _join("sk", "-", "live-", "0123456789abcdef"),
    _join("gh", "p_", "short"), _join("gh", "p_", _MIXED), _join("gh", "x_", _MIXED, _MIXED),
    "high_pressure_value and ghp_ alone",
    _join("github", "_pat_", "tooshort123"),
    _join("AK", "IA", "Z7Q2M4XKP9RTB3W"), _join("AK", "IA", "Z7Q2M4XKP9RTB3WDX"), _join("ak", "ia", "z7q2m4xkp9rtb3wd"),
    "ASIAPACIFIC regional office and ASIAN cuisine",
    "xoxo-gossip-girl-2021-forever", _join("xo", "xb-", "no-digits-here-at-all"), _join("xo", "xb-", "12"),
    "Zażółć gęślą jaźń: migracja bazy danych", "BILL-123 sk- ghp_ AKIA xoxb-",
])
def test_lookalikes_of_provider_tokens_are_left_alone(tmp_path, text):
    assert scrub_text(text, tmp_path / "home") == text


def test_bare_tokens_in_a_query_never_reach_the_embedder():
    from recall_embeddings import embedding_text
    query = "how do we ship kubernetes rollout token " + BARE_TOKENS["openai lowercase"]
    assert embedding_text(query) == "how do we ship kubernetes rollout token <redacted>"
