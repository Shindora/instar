"""Scriptable agent functions. Each calls the model through the base URL in its environment."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Literal

import httpx
from opentelemetry import trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import TracerProvider

type Provider = Literal["openai", "anthropic", "gemini"]

MODEL = "mock-model"
TIMEOUT_S = 30.0
IMPORT_TIME_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")


def _request(
    provider: Provider, prompt: str, *, stream: bool, base_url: str | None
) -> tuple[str, dict[str, str], dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    if provider == "gemini":
        base = base_url if base_url is not None else os.environ["GOOGLE_GEMINI_BASE_URL"]
        headers = {"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "test-key")}
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        return f"{base.rstrip('/')}/v1beta/models/{MODEL}:{method}", headers, body
    if provider == "openai":
        base = base_url if base_url is not None else os.environ["OPENAI_BASE_URL"]
        key = os.environ.get("OPENAI_API_KEY", "test-key")
        body = {"model": MODEL, "messages": messages, "stream": stream}
        return f"{base.rstrip('/')}/chat/completions", {"authorization": f"Bearer {key}"}, body
    base = base_url if base_url is not None else os.environ["ANTHROPIC_BASE_URL"]
    key = os.environ.get("ANTHROPIC_API_KEY", "test-key")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    body = {"model": MODEL, "max_tokens": 256, "messages": messages, "stream": stream}
    return f"{base.rstrip('/')}/v1/messages", headers, body


def _event_text(provider: Provider, event: dict[str, Any]) -> str:
    if provider == "openai":
        return "".join(
            (choice.get("message") or choice.get("delta") or {}).get("content") or ""
            for choice in event.get("choices", [])
        )
    if provider == "anthropic":
        if event.get("type") == "content_block_delta":
            return event["delta"]["text"]
        return "".join(block.get("text", "") for block in event.get("content", []))
    return "".join(
        part.get("text", "")
        for candidate in event.get("candidates", [])
        for part in candidate["content"]["parts"]
    )


def _reply(provider: Provider, response: httpx.Response, *, stream: bool) -> str:
    response.raise_for_status()
    if not stream:
        return _event_text(provider, response.json())
    lines = response.text.splitlines()
    docs = [line.removeprefix("data:").strip() for line in lines if line.startswith("data:")]
    return "".join(_event_text(provider, json.loads(doc)) for doc in docs if doc != "[DONE]")


def answer(
    prompt: str,
    *,
    calls: int = 1,
    provider: Provider = "openai",
    stream: bool = False,
    base_url: str | None = None,
) -> str:
    """Well-behaved: make `calls` model calls and return the last reply."""
    reply = ""
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for _ in range(calls):
            url, headers, body = _request(provider, prompt, stream=stream, base_url=base_url)
            reply = _reply(provider, client.post(url, headers=headers, json=body), stream=stream)
    return reply


async def answer_async(
    prompt: str,
    *,
    calls: int = 1,
    provider: Provider = "openai",
    stream: bool = False,
    base_url: str | None = None,
) -> str:
    """Async twin of `answer`."""
    reply = ""
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        for _ in range(calls):
            url, headers, body = _request(provider, prompt, stream=stream, base_url=base_url)
            response = await client.post(url, headers=headers, json=body)
            reply = _reply(provider, response, stream=stream)
    return reply


def looping(prompt: str, *, calls: int = 25, provider: Provider = "openai") -> str:
    """Stuck in a loop: asks the model the same thing over and over."""
    return answer(prompt, calls=calls, provider=provider)


def stalling(prompt: str, *, stall_s: float = 3600.0, provider: Provider = "openai") -> str:
    """Makes one call, then hangs."""
    reply = answer(prompt, provider=provider)
    time.sleep(stall_s)
    return reply


def raising(prompt: str, *, provider: Provider = "openai") -> str:
    """Makes one call, then raises."""
    answer(prompt, provider=provider)
    raise RuntimeError("agent failed on purpose")


def import_time_answer(prompt: str, *, calls: int = 1) -> str:
    """Uses the OpenAI base URL read once when this module was imported, like many SDK clients."""
    return answer(prompt, calls=calls, base_url=IMPORT_TIME_OPENAI_BASE_URL)


async def traced_answer(
    prompt: str,
    *,
    tools: Sequence[str] = ("lookup",),
    provider: Provider = "openai",
    tracer_provider: TracerProvider | None = None,
) -> str:
    """Emits GenAI spans: one agent span, then a tool span and a model span per tool."""
    tracer = trace.get_tracer(__name__, tracer_provider=tracer_provider)
    agent_attrs = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "mock-agent"}
    reply = ""
    with tracer.start_as_current_span("invoke_agent mock-agent", attributes=agent_attrs):
        for tool in tools:
            tool_attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool}
            with tracer.start_as_current_span(f"execute_tool {tool}", attributes=tool_attrs):
                pass
            chat_attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": MODEL}
            with tracer.start_as_current_span(
                f"chat {MODEL}", kind=trace.SpanKind.CLIENT, attributes=chat_attrs
            ):
                reply = await answer_async(prompt, provider=provider)
    return reply
