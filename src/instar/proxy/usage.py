from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """The provider's raw usage, flattened, plus one view across providers.

    `input_tokens` counts every input token, cached or not (the OpenTelemetry convention).
    `cached_input_tokens` were read from a cache; `cache_write_tokens` were written to one, of
    which `cache_write_1h_tokens` with a one-hour lifetime. `known` is False when the response
    carried no usage at all.
    """

    raw: dict[str, int]
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    output_tokens: int = 0
    response_model: str | None = None

    @property
    def known(self) -> bool:
        return bool(self.raw)


USAGE_KEYS = frozenset({"usage", "usageMetadata"})
GEMINI_METHODS = frozenset({"generateContent", "streamGenerateContent"})

usage_shapes: Registry[Callable[[dict[str, int]], Usage]] = Registry("usage shape")


@usage_shapes("openai")
def openai_usage(raw: dict[str, int]) -> Usage:
    """Prompt tokens already include the cached ones."""
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
    """Input tokens exclude cache reads and cache writes; both are reported beside them."""
    read = raw.get("cache_read_input_tokens", 0)
    written = raw.get("cache_creation_input_tokens", 0)
    one_hour = raw.get("cache_creation.ephemeral_1h_input_tokens", 0)
    return Usage(
        raw=raw,
        input_tokens=raw.get("input_tokens", 0) + read + written,
        cached_input_tokens=read,
        cache_write_tokens=written,
        cache_write_1h_tokens=min(one_hour, written),
        output_tokens=raw.get("output_tokens", 0),
    )


@usage_shapes("gemini")
def gemini_usage(raw: dict[str, int]) -> Usage:
    """The prompt count already includes cached content."""
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


def _model(doc: object) -> str | None:
    """The model the provider says answered: `model`, `message.model` or `modelVersion`."""
    if not isinstance(doc, dict):
        return None
    message = doc.get("message")
    nested = message.get("model") if isinstance(message, dict) else None
    for value in (doc.get("model"), doc.get("modelVersion"), nested):
        if isinstance(value, str) and value:
            return value
    return None


def parse_usage(path: str, body: bytes) -> Usage:
    """Read token usage and the answering model from a JSON body or an SSE stream."""
    raw: dict[str, int] = {}
    model = None
    text = body.decode("utf-8", "replace")
    docs = [line[5:] for line in text.splitlines() if line.startswith("data:")] or [text]
    for doc in docs:
        try:
            parsed = json.loads(doc)
        except ValueError:
            continue
        _collect(parsed, raw)
        model = model or _model(parsed)
    usage = usage_shapes[shape_of(path)](raw)
    return copy.replace(usage, response_model=model)
