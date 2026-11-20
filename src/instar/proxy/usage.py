from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """The provider's raw usage, flattened, plus the OpenTelemetry view where input counts cache."""

    raw: dict[str, int]
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


USAGE_KEYS = frozenset({"usage", "usageMetadata"})
GEMINI_METHODS = frozenset({"generateContent", "streamGenerateContent"})

usage_shapes: Registry[Callable[[dict[str, int]], Usage]] = Registry("usage shape")


@usage_shapes("openai")
def openai_usage(raw: dict[str, int]) -> Usage:
    return Usage(
        raw=raw,
        input_tokens=raw.get("prompt_tokens", raw.get("input_tokens", 0)),
        cached_input_tokens=raw.get(
            "prompt_tokens_details.cached_tokens", raw.get("input_tokens_details.cached_tokens", 0)
        ),
        output_tokens=raw.get("completion_tokens", raw.get("output_tokens", 0)),
    )


@usage_shapes("anthropic")
def anthropic_usage(raw: dict[str, int]) -> Usage:
    cached = raw.get("cache_read_input_tokens", 0)
    written = raw.get("cache_creation_input_tokens", 0)
    return Usage(
        raw=raw,
        input_tokens=raw.get("input_tokens", 0) + cached + written,
        cached_input_tokens=cached,
        output_tokens=raw.get("output_tokens", 0),
    )


@usage_shapes("gemini")
def gemini_usage(raw: dict[str, int]) -> Usage:
    return Usage(
        raw=raw,
        input_tokens=raw.get("promptTokenCount", 0),
        cached_input_tokens=raw.get("cachedContentTokenCount", 0),
        output_tokens=raw.get("candidatesTokenCount", 0) + raw.get("thoughtsTokenCount", 0),
    )


def shape_of(path: str) -> str:
    if path.rpartition(":")[2] in GEMINI_METHODS:
        return "gemini"
    return "anthropic" if path.rstrip("/").endswith("messages") else "openai"


def _flatten(usage: dict[str, object], out: dict[str, int], prefix: str = "") -> None:
    for key, value in usage.items():
        if isinstance(value, dict):
            _flatten(value, out, f"{prefix}{key}.")
        elif isinstance(value, int) and not isinstance(value, bool):
            name = f"{prefix}{key}"
            out[name] = max(out.get(name, 0), value)


def _collect(doc: object, out: dict[str, int]) -> None:
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key in USAGE_KEYS and isinstance(value, dict):
                _flatten(value, out)
            else:
                _collect(value, out)
    elif isinstance(doc, list):
        for value in doc:
            _collect(value, out)


def parse_usage(path: str, body: bytes) -> Usage:
    """Read token usage from a JSON body or an SSE stream, in either API shape."""
    raw: dict[str, int] = {}
    text = body.decode("utf-8", "replace")
    docs = [line[5:] for line in text.splitlines() if line.startswith("data:")] or [text]
    for doc in docs:
        try:
            _collect(json.loads(doc), raw)
        except ValueError:
            continue
    return usage_shapes[shape_of(path)](raw)
