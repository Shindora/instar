from __future__ import annotations

import asyncio
import json
import resource
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING

import httpx
import pytest

from instar.events import CallRole, LLMCall
from instar.proxy.app import Proxy
from instar.proxy.upstreams import judge_url, model_url, resolve_upstreams
from instar.store import JsonlSink
from tests.fixtures.agents import answer
from tests.fixtures.mock_upstream import serve

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from instar.events import Event
    from tests.fixtures.agents import Provider
    from tests.fixtures.mock_upstream import MockUpstream

PROVIDERS: list[Provider] = ["openai", "anthropic", "gemini"]


class ListSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    @property
    def calls(self) -> list[LLMCall]:
        return [e for e in self.events if isinstance(e, LLMCall)]


def build_proxy(
    upstream_url: str,
    sink: ListSink,
    *,
    seed: int = 0,
    fault_rates: Mapping[str, float] | None = None,
    spend_cap_usd: float | None = None,
    latency_s: float = 5.0,
) -> Proxy:
    return Proxy(
        upstreams=direct_bases(upstream_url),
        sink=sink,
        seed=seed,
        fault_rates=fault_rates,
        spend_cap_usd=spend_cap_usd,
        latency_s=latency_s,
    )


def direct_bases(upstream_url: str) -> dict[str, str]:
    return {"openai": f"{upstream_url}/v1", "anthropic": upstream_url, "gemini": upstream_url}


def call_for(upstream: str, *, stream: bool = False) -> tuple[str, dict[str, object]]:
    """Path under the upstream's base URL, and the JSON body, for one model call."""
    if upstream == "gemini":
        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return f"v1beta/models/m:{method}", {"contents": [{"parts": [{"text": "a b"}]}]}
    path = "chat/completions" if upstream == "openai" else "v1/messages"
    return path, {"model": "m", "messages": [{"role": "user", "content": "a b"}], "stream": stream}


async def run_session(
    client: httpx.AsyncClient,
    proxy_url: str,
    session: str,
    *,
    calls: int,
    upstream: str = "openai",
    stream: bool = False,
) -> list[int]:
    path, payload = call_for(upstream, stream=stream)
    url = f"{model_url(proxy_url, session, upstream)}/{path}"
    return [(await client.post(url, json=payload)).status_code for _ in range(calls)]


@pytest.fixture
def sink() -> ListSink:
    return ListSink()


@pytest.fixture
def proxy_url(upstream_url: str, sink: ListSink) -> Iterator[str]:
    with serve(build_proxy(upstream_url, sink).build_app()) as url:
        yield url


@pytest.mark.parametrize("upstream", PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
def test_pass_through_is_byte_identical(
    upstream_url: str, proxy_url: str, upstream: str, stream: bool
) -> None:
    path, payload = call_for(upstream, stream=stream)
    if upstream == "openai" and stream:
        payload["stream_options"] = {"include_usage": True}
    direct = httpx.post(f"{direct_bases(upstream_url)[upstream]}/{path}", json=payload)
    via = httpx.post(f"{model_url(proxy_url, 's1', upstream)}/{path}", json=payload)
    assert via.status_code == direct.status_code == 200
    assert via.content == direct.content
    assert via.headers["content-type"] == direct.headers["content-type"]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("stream", [False, True])
def test_usage_is_read_from_every_shape(
    mock_upstream: MockUpstream,
    sink: ListSink,
    proxy_url: str,
    provider: Provider,
    stream: bool,
) -> None:
    mock_upstream.cached_input_tokens = 5
    base_url = model_url(proxy_url, "s1", provider)
    assert answer("a b", provider=provider, stream=stream, base_url=base_url) == "echo: a b"
    [call] = sink.calls
    assert (call.input_tokens, call.cached_input_tokens, call.output_tokens) == (7, 5, 3)
    assert call.usage
    assert (call.session, call.upstream, call.index) == ("s1", provider, 1)
    assert call.model == "mock-model"
    assert call.started_ns <= call.ended_ns


def test_openai_stream_gets_usage_added(sink: ListSink, proxy_url: str) -> None:
    path, payload = call_for("openai", stream=True)
    text = httpx.post(f"{model_url(proxy_url, 's1', 'openai')}/{path}", json=payload).text
    assert '"usage"' in text
    assert sink.calls[0].output_tokens == 3


async def test_fault_schedule_is_identical_across_runs(upstream_url: str) -> None:
    rates = {"http_429": 0.2, "http_500": 0.1, "truncate": 0.1}

    async def run_once() -> dict[tuple[str, int], str | None]:
        sink = ListSink()
        with serve(build_proxy(upstream_url, sink, seed=7, fault_rates=rates).build_app()) as url:
            async with httpx.AsyncClient(timeout=30) as client:
                sessions = [
                    run_session(client, url, f"session-{i}", calls=4, upstream=PROVIDERS[i % 3])
                    for i in range(50)
                ]
                await asyncio.gather(*sessions)
        return {(c.session, c.index): c.fault for c in sink.calls}

    first = await run_once()
    second = await run_once()
    assert len(first) == 200
    assert first == second
    assert set(first.values()) == {"http_429", "http_500", "truncate", None}


@pytest.mark.parametrize("upstream", PROVIDERS)
async def test_sessions_with_one_fault_key_get_the_same_faults(
    upstream_url: str, upstream: str
) -> None:
    sink = ListSink()
    proxy = build_proxy(upstream_url, sink, fault_rates={"http_429": 0.3, "http_500": 0.2})
    proxy.fault_keys.update({"arm-a": "task-1:0", "arm-b": "task-1:0"})
    with serve(proxy.build_app()) as url:
        async with httpx.AsyncClient() as client:
            a = await run_session(client, url, "arm-a", calls=20, upstream=upstream)
            b = await run_session(client, url, "arm-b", calls=20, upstream=upstream)
    assert a == b
    assert set(a) == {200, 429, 500}


@pytest.mark.parametrize("upstream", PROVIDERS)
async def test_judge_calls_are_never_faulted_or_capped(upstream_url: str, upstream: str) -> None:
    sink = ListSink()
    proxy = build_proxy(upstream_url, sink, fault_rates={"http_429": 1.0}, spend_cap_usd=1.0)
    path, payload = call_for(upstream)
    with serve(proxy.build_app()) as url:
        agent = f"{model_url(url, 's1', upstream)}/{path}"
        judge = f"{judge_url(url, 's1', upstream)}/{path}"
        async with httpx.AsyncClient() as client:
            assert (await client.post(agent, json=payload)).status_code == 429
            assert (await client.post(judge, json=payload)).status_code == 200
            proxy.add_spend(1.0)
            capped = await client.post(agent, json=payload)
            assert (await client.post(judge, json=payload)).status_code == 200
    assert capped.json()["error"]["type"] == "budget_exceeded"
    assert proxy.spent_usd == 1.0
    assert [(c.role, c.fault, c.index) for c in sink.calls] == [
        (CallRole.AGENT, "http_429", 1),
        (CallRole.JUDGE, None, 1),
        (CallRole.AGENT, "budget", 2),
        (CallRole.JUDGE, None, 2),
    ]


async def test_spend_cap_rejects_once_reached(upstream_url: str) -> None:
    sink = ListSink()
    proxy = build_proxy(upstream_url, sink, spend_cap_usd=0.5)
    with serve(proxy.build_app()) as url:
        async with httpx.AsyncClient() as client:
            proxy.add_spend(0.4)
            assert await run_session(client, url, "s1", calls=1) == [200]
            proxy.add_spend(0.1)
            assert await run_session(client, url, "s1", calls=2) == [429, 429]
    assert [c.fault for c in sink.calls] == [None, "budget", "budget"]


@pytest.fixture
def many_open_files() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
    if soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


@pytest.mark.usefixtures("many_open_files")
async def test_300_concurrent_sessions_complete_without_errors(
    sink: ListSink, proxy_url: str
) -> None:
    limits = httpx.Limits(max_connections=300)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:
        results = await asyncio.gather(
            *(
                run_session(client, proxy_url, f"s{i}", calls=3, upstream=up, stream=i % 2 == 0)
                for i, up in zip(range(300), PROVIDERS * 100, strict=True)
            )
        )
    assert all(status == 200 for statuses in results for status in statuses)
    assert len(sink.calls) == 900
    assert all(c.status == 200 and c.error is None and c.output_tokens == 3 for c in sink.calls)


async def test_stopping_the_server_loses_no_events(
    upstream_url: str, mock_upstream: MockUpstream
) -> None:
    mock_upstream.delay_s = 0.5
    sink = ListSink()
    with ExitStack() as stack:
        url = stack.enter_context(serve(build_proxy(upstream_url, sink).build_app()))
        async with httpx.AsyncClient(timeout=10) as client:
            pending = [
                asyncio.create_task(run_session(client, url, f"s{i}", calls=1, stream=True))
                for i in range(20)
            ]
            await asyncio.sleep(0.2)
            await asyncio.to_thread(stack.close)
            results = await asyncio.gather(*pending)
    assert results == [[200]] * 20
    assert len(sink.calls) == 20


def test_unknown_upstream_and_sessionless_calls_are_recorded(
    sink: ListSink, proxy_url: str
) -> None:
    path, payload = call_for("openai")
    assert httpx.post(f"{proxy_url}/s/s1/nope/x", json={}).status_code == 404
    assert httpx.post(f"{proxy_url}/openai/{path}", json=payload).status_code == 200
    assert [(c.session, c.upstream, c.status) for c in sink.calls] == [
        ("s1", "nope", 404),
        ("", "openai", 200),
    ]


def test_declared_upstreams_beat_chained_env_which_beats_defaults() -> None:
    environ = {"OPENAI_BASE_URL": "http://gateway/v1", "MY_LLM_URL": "http://llm"}
    upstreams = resolve_upstreams(
        {"anthropic": "http://declared"}, {"MY_LLM_URL": "llm"}, environ=environ
    )
    assert upstreams == {
        "openai": "http://gateway/v1",
        "anthropic": "http://declared",
        "gemini": "https://generativelanguage.googleapis.com",
        "llm": "http://llm",
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_cli_proxy_logs_calls_and_survives_sigterm(upstream_url: str, tmp_path: Path) -> None:
    port, out = _free_port(), tmp_path / "calls.jsonl"
    command = [sys.executable, "-m", "instar.cli", "proxy", "--port", str(port)]
    command += ["--upstream", f"openai={upstream_url}/v1", "--out", str(out)]
    proc = subprocess.Popen(command, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                httpx.get(f"http://127.0.0.1:{port}/")
                break
            except httpx.ConnectError:
                assert time.monotonic() < deadline, "proxy did not start"
                time.sleep(0.05)
        path, payload = call_for("openai", stream=True)
        response = httpx.post(f"http://127.0.0.1:{port}/s/abc/openai/{path}", json=payload)
        assert response.status_code == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    [line] = out.read_text().splitlines()
    event = json.loads(line)
    assert (event["type"], event["session"], event["output_tokens"]) == ("LLMCall", "abc", 3)
    assert "a b" not in line


@pytest.mark.parametrize("upstream", PROVIDERS)
@pytest.mark.parametrize(
    ("fault", "status"), [("http_429", 429), ("http_500", 500), ("latency", 200), ("truncate", 200)]
)
async def test_every_fault_works_on_every_shape(
    upstream_url: str, mock_upstream: MockUpstream, upstream: str, fault: str, status: int
) -> None:
    mock_upstream.chunk_delay_s = 0.02
    path, payload = call_for(upstream, stream=True)
    sink = ListSink()
    proxy = build_proxy(upstream_url, sink, fault_rates={fault: 1.0}, latency_s=0.2)
    with serve(proxy.build_app()) as url:
        async with httpx.AsyncClient(timeout=10) as client:
            full = await client.post(f"{direct_bases(upstream_url)[upstream]}/{path}", json=payload)
            started = time.perf_counter()
            response = await client.post(f"{model_url(url, 's1', upstream)}/{path}", json=payload)
            elapsed_s = time.perf_counter() - started
    [call] = sink.calls
    assert (response.status_code, call.status, call.fault) == (status, status, fault)
    if fault == "truncate":
        assert 0 < len(response.content) < len(full.content)
    if fault == "latency":
        assert elapsed_s >= 0.2
        assert call.output_tokens == 3


def test_query_string_is_never_recorded(sink: ListSink, proxy_url: str, tmp_path: Path) -> None:
    path, payload = call_for("gemini")
    url = f"{model_url(proxy_url, 's1', 'gemini')}/{path}?key=secret-key-123"
    assert httpx.post(url, json=payload).status_code == 200
    [call] = sink.calls
    log = JsonlSink(tmp_path / "calls.jsonl")
    log.emit(call)
    log.close()
    assert call.path == "v1beta/models/m:generateContent"
    assert "secret-key-123" not in repr(call)
    assert "secret-key-123" not in (tmp_path / "calls.jsonl").read_text()


def test_vertex_exists_only_when_its_variable_is_set(upstream_url: str) -> None:
    assert "vertex" not in resolve_upstreams(environ={})
    sink = ListSink()
    upstreams = resolve_upstreams(environ={"GOOGLE_VERTEX_BASE_URL": upstream_url})
    path = "v1/projects/p/locations/us/publishers/google/models/gemini-x:streamGenerateContent"
    payload = {"contents": [{"parts": [{"text": "a b"}]}]}
    with serve(Proxy(upstreams=upstreams, sink=sink).build_app()) as url:
        response = httpx.post(f"{model_url(url, 's1', 'vertex')}/{path}?alt=sse", json=payload)
    assert response.status_code == 200
    [call] = sink.calls
    assert (call.upstream, call.model, call.input_tokens, call.output_tokens) == (
        "vertex",
        "gemini-x",
        2,
        3,
    )
