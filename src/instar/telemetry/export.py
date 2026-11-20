"""Forward the proxy's own model-call spans and metrics to an external OTLP collector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from instar.events import LLMCall
from instar.telemetry import semconv

if TYPE_CHECKING:
    from instar.events import Event, Sink

SECONDS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]
TOKENS = [4**i for i in range(14)]


class ExportingSink:
    """Passes every event on to `inner`; also exports each model call as a span plus metrics.

    Needs the `otel` extra. `close()` flushes; uvicorn runs no atexit hooks on SIGTERM.
    """

    def __init__(self, inner: Sink, endpoint: str) -> None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        base = endpoint.rstrip("/")
        resource = Resource.create({"service.name": "instar-proxy"})
        self.inner = inner
        self.tracers = TracerProvider(resource=resource)
        self.tracers.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces"))
        )
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=f"{base}/v1/metrics"))
        self.meters = MeterProvider(resource=resource, metric_readers=[reader])
        self.tracer = self.tracers.get_tracer("instar.proxy")
        meter = self.meters.get_meter("instar.proxy")
        self.duration = meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            explicit_bucket_boundaries_advisory=SECONDS,
        )
        self.tokens = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}", explicit_bucket_boundaries_advisory=TOKENS
        )

    def emit(self, event: Event) -> None:
        self.inner.emit(event)
        if isinstance(event, LLMCall):
            try:
                self._export(event)
            except Exception:
                return

    def _export(self, call: LLMCall) -> None:
        from opentelemetry import trace

        trace_hex, _, span_hex = call.session.partition("-")
        context = None
        if len(trace_hex) == 32 and len(span_hex) == 16:
            parent = trace.SpanContext(
                int(trace_hex, 16),
                int(span_hex, 16),
                is_remote=True,
                trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
            )
            context = trace.set_span_in_context(trace.NonRecordingSpan(parent))
        model = call.model or "unknown"
        dims = {
            semconv.OPERATION: "chat",
            semconv.PROVIDER: call.upstream,
            semconv.REQUEST_MODEL: model,
            semconv.FAULT: call.fault or "none",
        }
        span = self.tracer.start_span(
            f"chat {model}",
            context=context,
            kind=trace.SpanKind.CLIENT,
            start_time=call.started_ns,
            attributes={
                **dims,
                semconv.SESSION: call.session,
                semconv.CALL_INDEX: call.index,
                "http.response.status_code": call.status,
                semconv.INPUT_TOKENS: call.input_tokens,
                semconv.OUTPUT_TOKENS: call.output_tokens,
                semconv.CACHE_READ_TOKENS: call.cached_input_tokens,
            },
        )
        if call.cost_usd is not None:
            span.set_attribute(semconv.COST, call.cost_usd)
        if call.status >= 400:
            span.set_status(trace.StatusCode.ERROR)
        span.end(end_time=call.ended_ns)
        self.duration.record((call.ended_ns - call.started_ns) / 1e9, dims)
        self.tokens.record(call.input_tokens, {**dims, "gen_ai.token.type": "input"})
        self.tokens.record(call.output_tokens, {**dims, "gen_ai.token.type": "output"})

    def close(self) -> None:
        self.tracers.shutdown()
        self.meters.shutdown()
