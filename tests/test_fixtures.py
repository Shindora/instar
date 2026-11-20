from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.fixtures.agents import answer, answer_async, raising, traced_answer

if TYPE_CHECKING:
    from tests.fixtures.agents import Provider
    from tests.fixtures.mock_upstream import MockUpstream

PROVIDERS: list[Provider] = ["openai", "anthropic"]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.usefixtures("model_env")
def test_sync_agent_end_to_end(
    mock_upstream: MockUpstream, provider: Provider, stream: bool
) -> None:
    assert answer("hello  world", calls=3, provider=provider, stream=stream) == "echo: hello  world"
    assert mock_upstream.calls == 3


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.usefixtures("model_env")
async def test_async_agent_end_to_end(
    mock_upstream: MockUpstream, provider: Provider, stream: bool
) -> None:
    assert await answer_async("hi", calls=2, provider=provider, stream=stream) == "echo: hi"
    assert mock_upstream.calls == 2


@pytest.mark.usefixtures("model_env")
def test_failure_and_delay_switches(mock_upstream: MockUpstream) -> None:
    mock_upstream.fail_status = 429
    with pytest.raises(httpx.HTTPStatusError) as err:
        answer("hi")
    assert err.value.response.status_code == 429
    assert err.value.response.headers["retry-after"] == "0"

    mock_upstream.fail_status = None
    mock_upstream.delay_s = 0.2
    started = time.perf_counter()
    answer("hi")
    assert time.perf_counter() - started >= 0.2


@pytest.mark.usefixtures("model_env")
def test_raising_agent_calls_model_then_raises(mock_upstream: MockUpstream) -> None:
    with pytest.raises(RuntimeError, match="on purpose"):
        raising("hi")
    assert mock_upstream.calls == 1


def _sse_docs(body: str) -> list[dict[str, object]]:
    docs = [line.removeprefix("data:").strip() for line in body.splitlines()]
    return [json.loads(d) for d in docs if d.startswith("{")]


def test_openai_stream_reports_usage_only_when_asked(upstream_url: str) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "a b"}], "stream": True}
    url = f"{upstream_url}/v1/chat/completions"
    plain = httpx.post(url, json=body).text
    asked = httpx.post(url, json={**body, "stream_options": {"include_usage": True}}).text
    assert not any("usage" in doc for doc in _sse_docs(plain))
    usage = [doc["usage"] for doc in _sse_docs(asked) if "usage" in doc]
    assert usage == [
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    ]


def test_cached_tokens_follow_each_provider(upstream_url: str, mock_upstream: MockUpstream) -> None:
    mock_upstream.cached_input_tokens = 5
    messages = [{"role": "user", "content": "a b"}]
    openai = httpx.post(f"{upstream_url}/v1/chat/completions", json={"messages": messages}).json()
    anthropic = httpx.post(f"{upstream_url}/v1/messages", json={"messages": messages}).json()
    assert openai["usage"]["prompt_tokens"] == 7
    assert openai["usage"]["prompt_tokens_details"]["cached_tokens"] == 5
    assert anthropic["usage"]["input_tokens"] == 2
    assert anthropic["usage"]["cache_read_input_tokens"] == 5


def test_responses_are_deterministic(upstream_url: str) -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    first = httpx.post(f"{upstream_url}/v1/messages", json=body).content
    assert httpx.post(f"{upstream_url}/v1/messages", json=body).content == first


@pytest.mark.usefixtures("model_env")
async def test_traced_agent_emits_genai_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reply = await traced_answer("hi", tools=["lookup", "search"], tracer_provider=provider)
    assert reply == "echo: hi"
    assert [span.name for span in exporter.get_finished_spans()] == [
        "execute_tool lookup",
        "chat mock-model",
        "execute_tool search",
        "chat mock-model",
        "invoke_agent mock-agent",
    ]
