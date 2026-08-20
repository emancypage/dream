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

    home_text = str(home.expanduser()).rstrip("/")
    if home_text:
        value = value.replace(home_text, "<home>")
    return value
