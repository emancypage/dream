"""Provider-neutral model invocation types and schema validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


WORKDIR = Path.home() / ".cache" / "dream" / "workdir"


def ensure_workdir() -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    return WORKDIR


class ProviderError(RuntimeError):
    """A configured model provider could not complete a request."""


class SchemaValidationError(ProviderError):
    """A provider returned output that does not satisfy the requested schema."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    system_prompt: str
    prompt: str
    schema: dict[str, Any]
    model: str | None
    timeout_seconds: int
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    output: dict[str, Any]
    raw_result: str
    provider: str
    model: str | None
    usage: Usage | None
    duration_ms: int
    total_cost_usd: float | None = None

    # Compatibility properties for the existing reporting code.
    @property
    def input_tokens(self) -> int | None:
        return self.usage.input_tokens if self.usage else None

    @property
    def output_tokens(self) -> int | None:
        return self.usage.output_tokens if self.usage else None

    @property
    def cache_creation_tokens(self) -> int | None:
        return self.usage.cache_creation_tokens if self.usage else None

    @property
    def cache_read_tokens(self) -> int | None:
        return self.usage.cache_read_tokens if self.usage else None


class ModelProvider(Protocol):
    def generate_structured(self, request: GenerationRequest) -> GenerationResult: ...


def validate_output(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate the JSON-schema subset used by dream without extra dependencies."""
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise SchemaValidationError(f"{path}: expected object")
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaValidationError(f"{path}: missing required keys: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise SchemaValidationError(f"{path}: unexpected keys: {', '.join(extra)}")
        for key, child in properties.items():
            if key in value:
                validate_output(value[key], child, f"{path}.{key}")
    elif expected == "array":
        if not isinstance(value, list):
            raise SchemaValidationError(f"{path}: expected array")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_output(item, item_schema, f"{path}[{index}]")
    elif expected == "string":
        if not isinstance(value, str):
            raise SchemaValidationError(f"{path}: expected string")
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaValidationError(f"{path}: expected integer")
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SchemaValidationError(f"{path}: expected number")
    elif expected == "boolean":
        if not isinstance(value, bool):
            raise SchemaValidationError(f"{path}: expected boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: value {value!r} is not in enum")


# Old adapters and downstream code imported this name.
CallResult = GenerationResult
