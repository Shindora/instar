from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from instar.events import CallRole, LLMCall
from instar.proxy.faults import FaultContext, check_rates, error_response, faults, pick_fault
from instar.proxy.pricing import call_cost
from instar.proxy.usage import parse_usage

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Mapping, MutableMapping

    from starlette.requests import Request

    from instar.events import Sink
    from instar.proxy.pricing import Price

REQUEST_DROP = frozenset({"host", "content-length", "transfer-encoding", "connection"})
RESPONSE_DROP = frozenset({"content-length", "transfer-encoding", "connection"})
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
MODEL_IN_PATH = re.compile(r"models/([^/:]+)")


class Proxy:
    """LLM reverse proxy: logs every call, injects seeded faults and enforces the spend cap.

    Routes: /s/<session>/<upstream>/<path> for the agent, /j/<session>/<upstream>/<path> for the
    judge (never faulted, never capped), /<upstream>/<path> for calls that carry no session.
    Agent calls stop once `token_cap` tokens or `spend_cap_usd` of priced cost are used. Cost
    comes only from `prices`; a call to an unpriced model has no cost and says why. Call
    counters, tokens and spend live in memory: one run, one proxy.
    """

    def __init__(
        self,
        *,
        upstreams: Mapping[str, str],
        sink: Sink,
        seed: int = 0,
        fault_rates: Mapping[str, float] | None = None,
        latency_s: float = 5.0,
        spend_cap_usd: float | None = None,
        token_cap: int | None = None,
        prices: Mapping[str, Price] | None = None,
        traces: MutableMapping[str, str] | None = None,
    ) -> None:
        self.upstreams = dict(upstreams)
        self.sink = sink
        self.seed = seed
        self.fault_rates = dict(fault_rates or {})
        check_rates(self.fault_rates)
        self.latency_s = latency_s
        self.spend_cap_usd = spend_cap_usd
        self.spent_usd = 0.0
        self.token_cap = token_cap
        self.tokens_used = 0
        self.prices = dict(prices or {})
        self.traces = traces if traces is not None else {}
        self.fault_keys: dict[str, str] = {}
        self._calls: Counter[tuple[CallRole, str]] = Counter()
        self._client: httpx.AsyncClient | None = None
        self._drains: set[asyncio.Task[None]] = set()

    def add_spend(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

    def build_app(self) -> Starlette:
        @asynccontextmanager
        async def lifespan(_: Starlette) -> AsyncGenerator[None]:
            # ponytail: no pool cap, so calls never queue inside the proxy and skew latency.
            # httpcore's pool costs ~10 ms a call at 300 open connections under a burst; that
            # is noise next to model latency. Revisit if a run needs thousands of calls a second.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(None, connect=10.0),
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=200),
            ) as client:
                self._client = client
                yield
                await asyncio.gather(*self._drains, return_exceptions=True)
            self._client = None

        return Starlette(
            lifespan=lifespan,
            routes=[
                Route("/s/{session}/{upstream}/{path:path}", self._handle, methods=METHODS),
                Route("/j/{session}/{upstream}/{path:path}", self._handle, methods=METHODS),
                Route("/{upstream}/{path:path}", self._handle, methods=METHODS),
            ],
        )

    async def _handle(self, request: Request) -> Response:
        params = request.path_params
        role = CallRole.JUDGE if request.url.path.startswith("/j/") else CallRole.AGENT
        session: str = params.get("session", "")
        self._calls[role, session] += 1
        body = await request.body()
        model, body = _prepare_body(params["path"], body)
        call = LLMCall(
            session=session,
            role=role,
            upstream=params["upstream"],
            index=self._calls[role, session],
            method=request.method,
            path=params["path"],
            model=model,
            status=0,
            started_ns=time.time_ns(),
            ended_ns=0,
            traceparent=request.headers.get("traceparent"),
        )
        clock = time.perf_counter_ns()
        if session and call.traceparent:
            parts = call.traceparent.split("-")
            if len(parts) == 4 and len(parts[1]) == 32:
                self.traces.setdefault(parts[1], session)

        base_url = self.upstreams.get(call.upstream)
        if base_url is None:
            return self._reject(call, clock, error_response(404, "unknown_upstream"), None)
        max_chunks = None
        if role is CallRole.AGENT:
            if self.token_cap is not None and self.tokens_used >= self.token_cap:
                rejected = error_response(429, "token_cap_exceeded")
                return self._reject(call, clock, rejected, "token_cap")
            if self.spend_cap_usd is not None and self.spent_usd >= self.spend_cap_usd:
                return self._reject(call, clock, error_response(429, "budget_exceeded"), "budget")
            key = self.fault_keys.get(session, session)
            fault = pick_fault(self.fault_rates, seed=self.seed, key=key, index=call.index)
            if fault is not None:
                call = copy.replace(call, fault=fault)
                ctx = FaultContext(
                    session=session,
                    upstream=call.upstream,
                    index=call.index,
                    latency_s=self.latency_s,
                )
                action = await faults[fault](ctx)
                if isinstance(action, Response):
                    return self._reject(call, clock, action, fault)
                max_chunks = action.max_chunks

        return await self._forward(
            request, call, clock, f"{base_url.rstrip('/')}/{call.path}", body, max_chunks
        )

    async def _forward(
        self,
        request: Request,
        call: LLMCall,
        clock: int,
        url: str,
        body: bytes,
        max_chunks: int | None,
    ) -> Response:
        if self._client is None:
            raise RuntimeError("proxy app is not running; serve it with its lifespan")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in REQUEST_DROP}
        headers["accept-encoding"] = "identity"
        upstream_request = self._client.build_request(
            request.method, url, params=request.query_params, headers=headers, content=body
        )
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            failed = copy.replace(call, status=502, error=type(exc).__name__, cost_usd=0.0)
            self._emit(failed, clock)
            return JSONResponse(
                {"type": "error", "error": {"type": "proxy_upstream_error", "message": str(exc)}},
                502,
            )
        response_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() not in RESPONSE_DROP
        }
        return StreamingResponse(
            self._relay(call, clock, upstream, max_chunks),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    async def _relay(
        self, call: LLMCall, clock: int, upstream: httpx.Response, max_chunks: int | None
    ) -> AsyncIterator[bytes]:
        chunks: list[bytes] = []
        first_byte: list[int] = []
        source = upstream.aiter_raw()
        truncated = False
        try:
            async for chunk in source:
                if not first_byte:
                    first_byte.append(call.started_ns + time.perf_counter_ns() - clock)
                chunks.append(chunk)
                yield chunk
                if max_chunks is not None and len(chunks) >= max_chunks:
                    truncated = True
                    break
        except BaseException as exc:
            # ponytail: record before any await, so a cancelled relay still records its call.
            self._finish(call, clock, upstream, chunks, first_byte, error=type(exc).__name__)
            await upstream.aclose()
            raise
        if truncated:
            # The agent's response ends here; read the rest upstream so usage is still known.
            drain = asyncio.create_task(
                self._drain(call, clock, upstream, source, chunks, first_byte)
            )
            self._drains.add(drain)
            drain.add_done_callback(self._drains.discard)
            return
        self._finish(call, clock, upstream, chunks, first_byte, error=None)
        await upstream.aclose()

    async def _drain(
        self,
        call: LLMCall,
        clock: int,
        upstream: httpx.Response,
        source: AsyncIterator[bytes],
        chunks: list[bytes],
        first_byte: list[int],
    ) -> None:
        error = None
        try:
            async for chunk in source:
                chunks.append(chunk)
        except Exception as exc:  # the agent already has its truncated reply
            error = type(exc).__name__
        finally:
            self._finish(call, clock, upstream, chunks, first_byte, error=error)
            await upstream.aclose()

    def _finish(
        self,
        call: LLMCall,
        clock: int,
        upstream: httpx.Response,
        chunks: list[bytes],
        first_byte: list[int],
        *,
        error: str | None,
    ) -> None:
        usage = parse_usage(call.path, b"".join(chunks))
        cost = call_cost(self.prices, call.model, upstream.status_code, usage)
        if call.role is CallRole.AGENT:
            self.tokens_used += usage.input_tokens + usage.output_tokens
            self.spent_usd += cost.usd or 0.0
        finished = copy.replace(
            call,
            status=upstream.status_code,
            error=error,
            first_byte_ns=first_byte[0] if first_byte else None,
            usage=usage.raw,
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_write_1h_tokens=usage.cache_write_1h_tokens,
            output_tokens=usage.output_tokens,
            response_model=usage.response_model,
            usage_known=usage.known or upstream.status_code >= 400,
            cost_usd=cost.usd,
            cost_note=cost.note,
            cost_approximate=cost.approximate,
        )
        self._emit(finished, clock)

    def _reject(self, call: LLMCall, clock: int, response: Response, fault: str | None) -> Response:
        rejected = copy.replace(call, status=response.status_code, fault=fault, cost_usd=0.0)
        self._emit(rejected, clock)
        return response

    def _emit(self, call: LLMCall, clock: int) -> None:
        ended_ns = call.started_ns + time.perf_counter_ns() - clock
        self.sink.emit(copy.replace(call, ended_ns=ended_ns))


def _prepare_body(path: str, body: bytes) -> tuple[str | None, bytes]:
    """Find the model (body, or path for Gemini); make OpenAI chat streams report usage."""
    in_path = MODEL_IN_PATH.search(path)
    model = in_path[1] if in_path else None
    try:
        payload = json.loads(body) if body else None
    except ValueError:
        return model, body
    if not isinstance(payload, dict):
        return model, body
    model = payload.get("model", model)
    if (
        path.endswith("chat/completions")
        and payload.get("stream")
        and "stream_options" not in payload
    ):
        payload["stream_options"] = {"include_usage": True}
        body = json.dumps(payload).encode()
    return model if isinstance(model, str) else None, body
