"""Deterministic privacy scrubbing for untrusted recall text."""

from __future__ import annotations

import re
from pathlib import Path


_PEM = re.compile(
    r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", re.IGNORECASE | re.DOTALL
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_CONNECTION = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis)://[^\s<>]+",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?im)\b(?:[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)|"
    r"(?:AWS|GITHUB|OPENAI|ANTHROPIC)_[A-Z0-9_]+)\s*=\s*([^\s#;,]+)"
)
_KEY_VALUE = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd|token)"
    r"\s*[:=]\s*([^\s#;,]+)"
)
# Provider tokens without a label. Each needs its provider's prefix plus a
# realistic length and charset, so words and identifiers that merely share a
# prefix (sk-learn, xoxo, ASIAN) stay readable. Prefixed OpenAI keys also need
# an uppercase letter and a digit, because kebab-case names such as
# sk-admin-dashboard-settings-panel share the prefix and the charset.
_BARE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:proj|svcacct|admin|None)-(?=[a-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{40,}"  # OpenAI prefixed keys
    r"|sk-ant-[a-z]{3,5}\d{2}-[A-Za-z0-9_-]{20,}"  # Anthropic keys
    r"|sk-[A-Za-z0-9]{20,}[A-Za-z0-9_-]*"  # OpenAI legacy keys
    r"|gh[pousr]_[A-Za-z0-9]{30,}"  # GitHub tokens
    r"|github_pat_[A-Za-z0-9_]{40,}"  # GitHub fine-grained tokens
    # AWS access key IDs. An accepted trade-off: a rare all-caps constant of
    # exactly 20 letters and digits that starts with AKIA or ASIA matches too.
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"
    r"|xox[abeprs]-(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{10,}"  # Slack tokens
    r"|xapp-\d-[A-Za-z0-9-]{20,}"  # Slack app-level tokens
    r")"
)


def scrub_text(text: str, home: Path) -> str:
    """Replace credentials, private keys, and local home paths.

    The operation is deliberately deterministic and conservative: labels are
    retained where possible, while values are replaced by ``<redacted>``.
    """
    value = str(text)
    value = _PEM.sub("<redacted>", value)
    value = _BEARER.sub("Bearer <redacted>", value)
    value = _CONNECTION.sub("<redacted>", value)
    value = _ASSIGNMENT.sub(lambda m: m.group(0)[: m.group(0).find("=") + 1] + " <redacted>", value)
    value = _KEY_VALUE.sub(lambda m: m.group(0)[: m.group(0).find(m.group(1))] + "<redacted>", value)
    # Last, so a labelled value keeps its label and earlier output is unchanged.
    value = _BARE_TOKEN.sub("<redacted>", value)

    home_text = str(home.expanduser()).rstrip("/")
    if home_text:
        value = value.replace(home_text, "<home>")
    return value
