from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Message:
    seq: int
    role: str
    timestamp: str | None
    text: str


@dataclass(frozen=True)
class TranscriptRef:
    source: str
    external_session_id: str
    path: Path
    revision: str


@dataclass(frozen=True)
class ParsedSession:
    source: str
    external_session_id: str
    revision: str
    parser_version: str
    path: Path
    project_slug: str
    started_at: str | None
    ended_at: str | None
    cwd: str | None
    git_branch: str | None
    messages: list[Message]

    @property
    def total_chars(self) -> int:
        return sum(len(message.text) for message in self.messages)

    @property
    def user_msg_count(self) -> int:
        return sum(message.role == "user" for message in self.messages)

    @property
    def asst_msg_count(self) -> int:
        return sum(message.role == "assistant" for message in self.messages)


class TranscriptSource(Protocol):
    def discover(self) -> Iterable[TranscriptRef]: ...
    def parse(self, ref: TranscriptRef) -> ParsedSession | None: ...
