from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from starlette.requests import Request
    from starlette.responses import Response

MESSAGE_ID = "mock-0001"


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return ""


def _sse(data: object, event: str | None = None) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data)}\n\n".encode()


def _deltas(reply: str) -> list[str]:
    first, *rest = reply.split(" ")
    return [first, *(f" {word}" for word in rest)]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Turn:
    model: str
    reply: str
    input_tokens: int

    @property
    def output_tokens(self) -> int:
        return len(_deltas(self.reply))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> _Turn:
        messages = payload.get("messages") or []
        texts = [_text(m.get("content")) for m in messages]
        prompt = texts[-1] if texts else ""
        texts.append(_text(payload.get("system")))
        return cls(
            model=payload.get("model") or "mock-model",
            reply=f"echo: {prompt}",
            input_tokens=sum(len(t.split()) for t in texts),
        )

    @classmethod
    def from_gemini(cls, payload: dict[str, Any], model: str) -> _Turn:
        texts = [_text(c.get("parts")) for c in payload.get("contents") or []]
        prompt = texts[-1] if texts else ""
        texts.append(_text((payload.get("systemInstruction") or {}).get("parts")))
        return cls(
            model=model,
            reply=f"echo: {prompt}",
            input_tokens=sum(len(t.split()) for t in texts),
        )


@dataclass(kw_only=True)
class MockUpstream:
    """A fake model API (OpenAI, Anthropic, Gemini) with switches a test flips while it serves.

    Responses are deterministic, so a test can compare bytes with and without a proxy in between.
    """

    delay_s: float = 0.0
    chunk_delay_s: float = 0.0
    fail_status: int | None = None
    cached_input_tokens: int = 0
    calls: int = 0

    def build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/chat/completions", self._openai, methods=["POST"]),
                Route("/v1/messages", self._anthropic, methods=["POST"]),
                Route("/{prefix:path}/models/{target}", self._gemini, methods=["POST"]),
            ]
        )

    async def _receive(self, request: Request) -> dict[str, Any]:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return await request.json()

    def _stream(self, chunks: AsyncIterator[bytes]) -> StreamingResponse:
        async def paced() -> AsyncIterator[bytes]:
            async for chunk in chunks:
                yield chunk
                if self.chunk_delay_s:
                    await asyncio.sleep(self.chunk_delay_s)

        return StreamingResponse(paced(), media_type="text/event-stream")

    def _failure(self, body: dict[str, Any]) -> Response | None:
        if self.fail_status is None:
            return None
        headers = {"retry-after": "0"} if self.fail_status == 429 else None
        return JSONResponse(body, self.fail_status, headers=headers)

    async def _openai(self, request: Request) -> Response:
        payload = await self._receive(request)
        error = {"error": {"type": "mock_error", "message": "failure switch is on", "code": None}}
        if failure := self._failure(error):
            return failure
        turn = _Turn.from_payload(payload)
        prompt_tokens = turn.input_tokens + self.cached_input_tokens
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": turn.output_tokens,
            "total_tokens": prompt_tokens + turn.output_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_input_tokens},
        }
        if not payload.get("stream"):
            return JSONResponse(
                {
                    "id": MESSAGE_ID,
                    "object": "chat.completion",
                    "created": 0,
                    "model": turn.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": turn.reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            )
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))
        return self._stream(_openai_stream(turn, usage if include_usage else None))

    async def _anthropic(self, request: Request) -> Response:
        payload = await self._receive(request)
        error = {
            "type": "error",
            "error": {"type": "mock_error", "message": "failure switch is on"},
        }
        if failure := self._failure(error):
            return failure
        turn = _Turn.from_payload(payload)
        usage = {
            "input_tokens": turn.input_tokens,
            "output_tokens": turn.output_tokens,
            "cache_read_input_tokens": self.cached_input_tokens,
            "cache_creation_input_tokens": 0,
        }
        if not payload.get("stream"):
            return JSONResponse(
                {
                    "id": MESSAGE_ID,
                    "type": "message",
                    "role": "assistant",
                    "model": turn.model,
                    "content": [{"type": "text", "text": turn.reply}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": usage,
                }
            )
        return self._stream(_anthropic_stream(turn, usage))

    async def _gemini(self, request: Request) -> Response:
        payload = await self._receive(request)
        code = self.fail_status
        error = {"error": {"code": code, "message": "failure switch is on", "status": "MOCK"}}
        if failure := self._failure(error):
            return failure
        model, _, method = request.path_params["target"].partition(":")
        if method not in {"generateContent", "streamGenerateContent"}:
            return JSONResponse({"error": {"code": 404, "message": method}}, 404)
        turn = _Turn.from_gemini(payload, model)
        if method == "generateContent":
            return JSONResponse(self._gemini_chunk(turn, turn.reply, last=True))
        deltas = _deltas(turn.reply)
        chunks = [
            self._gemini_chunk(turn, delta, last=i == len(deltas) - 1)
            for i, delta in enumerate(deltas)
        ]
        if request.query_params.get("alt") != "sse":
            return JSONResponse(chunks)
        return self._stream(_sse_each(chunks))

    def _gemini_chunk(self, turn: _Turn, text: str, *, last: bool) -> dict[str, Any]:
        """One response document. Like the real API, only the last one carries usage."""
        content = {"role": "model", "parts": [{"text": text}]}
        candidate: dict[str, Any] = {"content": content, "index": 0}
        chunk: dict[str, Any] = {"candidates": [candidate], "modelVersion": turn.model}
        chunk["responseId"] = MESSAGE_ID
        if not last:
            return chunk
        candidate["finishReason"] = "STOP"
        prompt_tokens = turn.input_tokens + self.cached_input_tokens
        chunk["usageMetadata"] = {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": turn.output_tokens,
            "totalTokenCount": prompt_tokens + turn.output_tokens,
            "cachedContentTokenCount": self.cached_input_tokens,
        }
        return chunk


async def _sse_each(docs: list[dict[str, Any]]) -> AsyncIterator[bytes]:
    for doc in docs:
        yield _sse(doc)


async def _openai_stream(turn: _Turn, usage: dict[str, Any] | None) -> AsyncIterator[bytes]:
    base = {"id": MESSAGE_ID, "object": "chat.completion.chunk", "created": 0, "model": turn.model}
    for i, delta in enumerate(_deltas(turn.reply)):
        content = {"role": "assistant", "content": delta} if i == 0 else {"content": delta}
        yield _sse({**base, "choices": [{"index": 0, "delta": content, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage is not None:
        yield _sse({**base, "choices": [], "usage": usage})
    yield b"data: [DONE]\n\n"


async def _anthropic_stream(turn: _Turn, usage: dict[str, Any]) -> AsyncIterator[bytes]:
    message = {
        "id": MESSAGE_ID,
        "type": "message",
        "role": "assistant",
        "model": turn.model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**usage, "output_tokens": 1},
    }
    yield _sse({"type": "message_start", "message": message}, "message_start")
    block = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield _sse(block, "content_block_start")
    for delta in _deltas(turn.reply):
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": delta},
        }
        yield _sse(event, "content_block_delta")
    yield _sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
    end = {"stop_reason": "end_turn", "stop_sequence": None}
    final = {
        "type": "message_delta",
        "delta": end,
        "usage": {"output_tokens": usage["output_tokens"]},
    }
    yield _sse(final, "message_delta")
    yield _sse({"type": "message_stop"}, "message_stop")


@contextmanager
def serve(app: Starlette) -> Generator[str]:
    """Serve an app on a free loopback port from a background thread and yield its base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_keep_alive=120))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("server thread exited before it started")
            time.sleep(0.005)
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join()
        sock.close()
