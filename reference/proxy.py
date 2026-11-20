"""Level-0 LLM proxy for agent stress tests.

Point the agent's base URL at  http://<host>:8080/s/<session_id>  and every LLM
call is logged (tokens, cost, time to first byte, latency) to a JSONL file.
Faults and a spend cap are switched on with environment variables.

    pip install starlette httpx uvicorn
    UPSTREAM=https://api.anthropic.com uvicorn proxy:app --port 8080 --timeout-keep-alive 120

Keep the long keep-alive: agents sit idle between calls while tools run, and a
short timeout makes the proxy drop connections the agent is about to reuse.
Run a single worker: call counters and the spend cap live in process memory.

Telemetry (optional): set OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318)
and each call is also exported as an OpenTelemetry span plus metrics.

    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

The proxy forwards the agent's own auth headers untouched and never logs them
or the request/response bodies.
"""
import asyncio
import contextlib
import json
import os
import random
import time

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("UPSTREAM", "https://api.openai.com").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "llm_calls.jsonl")
SEED = os.environ.get("SEED", "0")
BUDGET_USD = float(os.environ.get("BUDGET_USD", "inf"))
# Probability per call, e.g. {"http_429": 0.1, "http_500": 0.02, "latency": 0.1, "latency_s": 5, "truncate": 0.05}
FAULTS = json.loads(os.environ.get("FAULTS", "{}"))
# USD per 1M input / output tokens, e.g. {"my-model": [3.0, 15.0]}
PRICES = json.loads(os.environ.get("PRICES", "{}"))

REQ_DROP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}
RESP_DROP = {"content-length", "transfer-encoding", "connection"}
FAULT_ORDER = ("http_429", "http_500", "latency", "truncate")

client = httpx.AsyncClient(
    timeout=httpx.Timeout(None, connect=10.0),  # agent calls can stream for minutes
    limits=httpx.Limits(max_connections=None, max_keepalive_connections=200),
)
counters: dict[str, int] = {}
spent = 0.0


OTEL = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
if OTEL:
    import hashlib
    import re

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    os.environ.setdefault("OTEL_SERVICE_NAME", "instar-proxy")
    PROVIDER = os.environ.get("GEN_AI_PROVIDER") or next(
        (p for p in ("anthropic", "openai") if p in UPSTREAM), "unknown")
    _resource = Resource.create()  # honours OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES
    _tracers = TracerProvider(resource=_resource)
    _tracers.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    _meters = MeterProvider(resource=_resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    tracer = _tracers.get_tracer("instar.proxy")
    meter = _meters.get_meter("instar.proxy")

    # The GenAI conventions are still "Development" and metric names are being renamed
    # upstream, so every name lives here. Default SDK buckets are sized for milliseconds.
    SECONDS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
    TOKENS = [4 ** i for i in range(14)]  # 1 .. 67M
    m_duration = meter.create_histogram(
        "gen_ai.client.operation.duration", unit="s", explicit_bucket_boundaries_advisory=SECONDS)
    m_first_chunk = meter.create_histogram(
        "gen_ai.client.operation.time_to_first_chunk", unit="s", explicit_bucket_boundaries_advisory=SECONDS)
    m_tokens = meter.create_histogram(
        "gen_ai.client.token.usage", unit="{token}", explicit_bucket_boundaries_advisory=TOKENS)
    m_cost = meter.create_counter("instar.cost", unit="{USD}")

    def parent_context(rec: dict):
        """Where the span hangs: the agent's own trace, else ids carried in the session, else a hash of it."""
        if rec.get("traceparent"):
            return TraceContextTextMapPropagator().extract({"traceparent": rec["traceparent"]})
        ids = re.fullmatch(r"([0-9a-f]{32})-([0-9a-f]{16})", rec["session"])
        if ids:
            trace_id, span_id = int(ids[1], 16), int(ids[2], 16)
        else:  # no driver yet: all calls of one session still land in one trace
            digest = hashlib.sha256(rec["session"].encode()).digest()
            trace_id, span_id = int.from_bytes(digest[:16], "big"), int.from_bytes(digest[16:24], "big")
        parent = trace.SpanContext(trace_id, span_id, is_remote=True, trace_flags=trace.TraceFlags(1))
        return trace.set_span_in_context(trace.NonRecordingSpan(parent))

    def emit(rec: dict) -> None:
        usage = rec.get("usage", {})
        model = rec["model"] or "unknown"
        operation = "embeddings" if "embeddings" in rec["path"] else "chat"
        failed = rec["status"] >= 400
        dims = {
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": PROVIDER,
            "gen_ai.request.model": model,
            "instar.fault": rec["fault"] or "none",
        }
        if failed:
            dims["error.type"] = str(rec["status"])
        # The convention counts cached tokens as input; Anthropic reports them separately.
        cached = usage.get("cache_read_input_tokens", 0)
        tokens_in = rec.get("tokens_in", 0) + cached + usage.get("cache_creation_input_tokens", 0)
        tokens_out = rec.get("tokens_out", 0)
        end_ns = time.time_ns()
        start_ns = min(int(rec["ts"] * 1e9), end_ns)  # "ts" is rounded, never let it pass the end

        span = tracer.start_span(
            f"{operation} {model}", context=parent_context(rec), kind=trace.SpanKind.CLIENT, start_time=start_ns,
            attributes={
                **dims,
                "session.id": rec["session"],  # high cardinality: spans only, never a metric label
                "instar.call_index": rec["n"],
                "http.response.status_code": rec["status"],
                "gen_ai.usage.input_tokens": tokens_in,
                "gen_ai.usage.output_tokens": tokens_out,
                "gen_ai.usage.cache_read.input_tokens": cached,
                "instar.cost_usd": rec.get("cost_usd", 0.0),
            },
        )
        if "ttfb_s" in rec:
            span.set_attribute("gen_ai.response.time_to_first_chunk", rec["ttfb_s"])
            m_first_chunk.record(rec["ttfb_s"], dims)
        if failed:
            span.set_status(trace.StatusCode.ERROR)
        span.end(end_time=end_ns)

        m_duration.record((end_ns - start_ns) / 1e9, dims)
        m_tokens.record(tokens_in, {**dims, "gen_ai.token.type": "input"})
        m_tokens.record(tokens_out, {**dims, "gen_ai.token.type": "output"})
        m_cost.add(rec.get("cost_usd", 0.0), dims)


def log(rec: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if OTEL:
        try:
            emit(rec)
        except Exception as exc:  # telemetry must never break the call being measured
            print(f"otel emit failed: {exc!r}")


def pick_fault(session: str, n: int) -> str | None:
    """Same (seed, session, call index) always gets the same fault, whatever the interleaving."""
    rng = random.Random(f"{SEED}:{session}:{n}")
    for name in FAULT_ORDER:
        if rng.random() < FAULTS.get(name, 0):
            return name
    return None


def find_usage(obj, out: dict) -> None:
    """Collect integers under any "usage" key; covers OpenAI and Anthropic response shapes."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "usage" and isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, int):
                        out[k] = max(out.get(k, 0), v)
            else:
                find_usage(val, out)
    elif isinstance(obj, list):
        for val in obj:
            find_usage(val, out)


def parse_usage(body: bytes) -> dict:
    usage: dict = {}
    text = body.decode("utf-8", "replace")
    docs = [line[5:] for line in text.splitlines() if line.startswith("data:")] or [text]
    for doc in docs:
        try:
            find_usage(json.loads(doc), usage)
        except ValueError:
            pass  # "[DONE]" or a non-JSON body
    return usage


def reject(rec: dict, status: int, kind: str) -> JSONResponse:
    rec.update(status=status, total_s=0.0)
    log(rec)
    body = {"type": "error", "error": {"type": kind, "message": f"injected by proxy: {kind}"}}
    return JSONResponse(body, status, headers={"retry-after": "1"})


async def handle(request):
    session = request.path_params.get("session", "default")
    path = request.path_params["path"]
    n = counters[session] = counters.get(session, 0) + 1
    body = await request.body()
    rec = {"ts": round(time.time(), 3), "session": session, "n": n, "path": path, "model": None, "fault": None}
    if "traceparent" in request.headers:
        rec["traceparent"] = request.headers["traceparent"]

    try:
        payload = json.loads(body) if body else {}
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        rec["model"] = payload.get("model")
        # OpenAI chat streams only report token usage when asked to.
        if path.endswith("chat/completions") and payload.get("stream") and "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
            body = json.dumps(payload).encode()

    if spent >= BUDGET_USD:
        rec["fault"] = "budget"
        return reject(rec, 429, "budget_exceeded")
    fault = rec["fault"] = pick_fault(session, n)
    if fault == "http_429":
        return reject(rec, 429, "rate_limit_error")
    if fault == "http_500":
        return reject(rec, 500, "api_error")
    if fault == "latency":
        await asyncio.sleep(FAULTS.get("latency_s", 5))

    headers = {k: v for k, v in request.headers.items() if k.lower() not in REQ_DROP}
    headers["accept-encoding"] = "identity"  # keep the body readable for usage parsing
    t0 = time.perf_counter()
    try:
        upstream = await client.send(
            client.build_request(
                request.method, f"{UPSTREAM}/{path}",
                params=request.query_params, headers=headers, content=body,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        rec.update(status=502, total_s=round(time.perf_counter() - t0, 3), error=type(exc).__name__)
        log(rec)
        return JSONResponse({"type": "error", "error": {"type": "proxy_upstream_error", "message": str(exc)}}, 502)

    async def relay():
        global spent
        chunks = []
        try:
            async for chunk in upstream.aiter_raw():
                if "ttfb_s" not in rec:
                    rec["ttfb_s"] = round(time.perf_counter() - t0, 3)
                chunks.append(chunk)
                yield chunk
                if fault == "truncate":
                    break  # cut the response after its first chunk
        finally:
            await upstream.aclose()
            usage = parse_usage(b"".join(chunks))
            tok_in = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            tok_out = usage.get("completion_tokens", usage.get("output_tokens", 0))
            price_in, price_out = PRICES.get(rec["model"] or "", (0, 0))
            cost = (tok_in * price_in + tok_out * price_out) / 1e6
            spent += cost
            rec.update(
                status=upstream.status_code, total_s=round(time.perf_counter() - t0, 3),
                tokens_in=tok_in, tokens_out=tok_out, cost_usd=round(cost, 6), usage=usage,
            )
            log(rec)

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in RESP_DROP}
    return StreamingResponse(relay(), status_code=upstream.status_code, headers=resp_headers)


@contextlib.asynccontextmanager
async def lifespan(app):
    yield
    await client.aclose()
    if OTEL:  # flush what is still buffered; uvicorn does not run atexit hooks on SIGTERM
        _tracers.shutdown()
        _meters.shutdown()


METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
app = Starlette(lifespan=lifespan, routes=[
    Route("/s/{session}/{path:path}", handle, methods=METHODS),
    Route("/{path:path}", handle, methods=METHODS),
])
