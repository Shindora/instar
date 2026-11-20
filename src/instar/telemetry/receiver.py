"""OTLP over HTTP (protobuf). Traces are normalised and joined to executions; metrics and logs
are accepted and dropped. A request is only queued, so an exporting agent never waits on us."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import zlib
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from instar.events import Span
from instar.telemetry import semconv

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable, Mapping, MutableMapping

    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.trace.v1.trace_pb2 import Span as OtlpSpan
    from starlette.requests import Request

    from instar.events import Sink
    from instar.telemetry.normalise import TelemetryMap

SESSION_FORMAT = re.compile(r"^[0-9a-f]{32}-[0-9a-f]{16}$")
MAX_BODY_BYTES = 16 * 1024 * 1024
ENCODINGS = frozenset({"", "identity", "gzip"})
STATUS_ERROR = 2
EMPTY_RESPONSE = b""  # an empty Export*ServiceResponse serialises to no bytes


def available() -> bool:
    """The receiver needs `opentelemetry-proto`, from the `otel` extra."""
    try:
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2  # noqa: F401
    except ImportError:
        return False
    return True


def _value(any_value: AnyValue) -> object:
    which = any_value.WhichOneof("value")
    if which is None or which in {"kvlist_value", "bytes_value"}:
        return None
    if which == "array_value":
        return [_value(v) for v in any_value.array_value.values]
    return getattr(any_value, which)


def _attributes(pairs: Iterable[KeyValue]) -> dict[str, object]:
    return {kv.key: _value(kv.value) for kv in pairs}


def _gunzip(body: bytes, limit: int) -> bytes | None:
    """Decompress, refusing anything that inflates past `limit`."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = inflater.decompress(body, limit + 1)
    except zlib.error:
        return None
    return None if len(data) > limit or inflater.unconsumed_tail else data


class Receiver:
    """Writes `Span` events to the sink. `traces` maps trace ids to sessions: the runner adds
    each session's own trace id and the proxy adds trace ids it sees in `traceparent`."""

    def __init__(
        self, *, sink: Sink, mapping: TelemetryMap, traces: MutableMapping[str, str]
    ) -> None:
        self.sink = sink
        self.mapping = mapping
        self.traces = traces
        self.received = 0
        self.unjoined = 0
        self.rejected = 0
        self._pending: asyncio.Queue[tuple[bytes, bool]] | None = None

    def build_app(self) -> Starlette:
        @contextlib.asynccontextmanager
        async def lifespan(_: Starlette) -> AsyncGenerator[None]:
            self._pending = asyncio.Queue()
            consumer = asyncio.create_task(self._consume(self._pending))
            yield
            await self._pending.join()
            consumer.cancel()

        return Starlette(
            lifespan=lifespan,
            routes=[
                Route("/v1/traces", self._traces, methods=["POST"]),
                Route("/v1/metrics", self._dropped, methods=["POST"]),
                Route("/v1/logs", self._dropped, methods=["POST"]),
            ],
        )

    async def _read(self, request: Request) -> tuple[bytes, bool] | Response:
        """The body and whether it is gzip, or the error response to send instead."""
        encoding = request.headers.get("content-encoding", "").strip().lower()
        if "protobuf" not in request.headers.get("content-type", "") or encoding not in ENCODINGS:
            return PlainTextResponse("instar reads OTLP/HTTP protobuf, plain or gzip", 415)
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return PlainTextResponse("export too large", 413)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                return PlainTextResponse("export too large", 413)
        return bytes(body), encoding == "gzip"

    async def _dropped(self, request: Request) -> Response:
        read = await self._read(request)
        if isinstance(read, Response):
            return read
        return Response(EMPTY_RESPONSE, media_type="application/x-protobuf")

    async def _traces(self, request: Request) -> Response:
        read = await self._read(request)
        if isinstance(read, Response):
            return read
        if self._pending is None:
            return PlainTextResponse("receiver is not running", 503)
        self._pending.put_nowait(read)
        return Response(EMPTY_RESPONSE, media_type="application/x-protobuf")

    async def _consume(self, pending: asyncio.Queue[tuple[bytes, bool]]) -> None:
        while True:
            body, gzipped = await pending.get()
            try:
                self.ingest(body, gzipped=gzipped)
            finally:
                pending.task_done()

    def ingest(self, body: bytes, *, gzipped: bool = False) -> None:
        from google.protobuf.message import DecodeError
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )

        data = _gunzip(body, MAX_BODY_BYTES) if gzipped else body
        message = ExportTraceServiceRequest()
        try:
            if data is None:
                raise DecodeError("bad gzip")
            message.ParseFromString(data)
        except DecodeError:
            self.rejected += 1
            return
        for resource_spans in message.resource_spans:
            resource = _attributes(resource_spans.resource.attributes)
            for scope_spans in resource_spans.scope_spans:
                for raw in scope_spans.spans:
                    self._emit(raw, resource)

    def session_for(self, trace_id: str, *attribute_sets: Mapping[str, object]) -> str:
        for attributes in attribute_sets:
            hint = attributes.get(semconv.SESSION)
            if isinstance(hint, str) and SESSION_FORMAT.match(hint):
                return hint
        return self.traces.get(trace_id, "")

    def _emit(self, raw: OtlpSpan, resource: Mapping[str, object]) -> None:
        attributes = _attributes(raw.attributes)
        trace_id = raw.trace_id.hex()
        session = self.session_for(trace_id, attributes, resource)
        self.received += 1
        self.unjoined += not session
        kept = self.mapping.keep(attributes)
        arguments = attributes.get(self.mapping.tool_arguments)
        if arguments is not None:
            digest = hashlib.sha256(str(arguments).encode()).hexdigest()[:16]
            kept[semconv.TOOL_ARGUMENTS_HASH] = digest
        self.sink.emit(
            Span(
                session=session,
                trace_id=trace_id,
                span_id=raw.span_id.hex(),
                parent_id=raw.parent_span_id.hex() or None,
                kind=self.mapping.kind(attributes),
                name=raw.name,
                started_ns=raw.start_time_unix_nano,
                ended_ns=raw.end_time_unix_nano,
                error=raw.status.code == STATUS_ERROR,
                attributes=kept,
            )
        )
