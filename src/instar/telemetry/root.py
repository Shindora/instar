"""Put an execution's spans in its session's trace. The session is a W3C trace id and span id:
spans `run` starts become children of that span, spans `judge` starts become children of a
second span derived from it, so judge spans can be told apart from the agent's."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from instar.telemetry import semconv

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from instar.events import Span


def judge_span_id(session: str) -> str:
    return hashlib.sha256(f"{session}:judge".encode()).hexdigest()[:16]


def root_span_id(session: str, phase: str) -> str:
    return judge_span_id(session) if phase == "judge" else session.partition("-")[2]


def install_provider() -> None:
    """Export the agent's spans to instar when the agent set up no tracer provider itself."""
    if not os.environ.get(semconv.TRACES_ENDPOINT):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


@contextmanager
def phase(session: str, name: str) -> Generator[None]:
    """Within the block the current span is the phase's non-recording root."""
    try:
        from opentelemetry import context, trace
    except ImportError:
        yield
        return
    root = trace.NonRecordingSpan(
        trace.SpanContext(
            int(session.partition("-")[0], 16),
            int(root_span_id(session, name), 16),
            is_remote=True,
            trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
        )
    )
    token = context.attach(trace.set_span_in_context(root))
    try:
        yield
    finally:
        context.detach(token)


def flush() -> None:
    """Export what the tracer provider still buffers; uvicorn and workers run no atexit hooks."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    force_flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if callable(force_flush):
        force_flush()


def split_phases(spans: Sequence[Span], session: str) -> tuple[list[Span], list[Span]]:
    """(agent spans, judge spans): a span whose parents lead to the judge's root is the judge's."""
    judge_root = judge_span_id(session)
    parents = {span.span_id: span.parent_id for span in spans}

    def judged(span_id: str | None) -> bool:
        seen: set[str] = set()
        while span_id is not None and span_id not in seen:
            if span_id == judge_root:
                return True
            seen.add(span_id)
            span_id = parents.get(span_id)
        return False

    agent = [span for span in spans if not judged(span.parent_id)]
    judge = [span for span in spans if judged(span.parent_id)]
    return agent, judge
