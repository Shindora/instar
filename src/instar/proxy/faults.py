from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse

from instar.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from starlette.responses import Response


@dataclass(frozen=True, slots=True, kw_only=True)
class FaultContext:
    session: str
    upstream: str
    index: int
    latency_s: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Forward:
    """Send the call upstream; relay at most `max_chunks` chunks of the response."""

    max_chunks: int | None = None


type FaultFn = Callable[[FaultContext], Awaitable[Response | Forward]]

faults: Registry[FaultFn] = Registry("fault")


def error_response(status: int, kind: str) -> JSONResponse:
    body = {"type": "error", "error": {"type": kind, "message": f"injected by instar: {kind}"}}
    return JSONResponse(body, status, headers={"retry-after": "1"})


@faults("http_429")
async def http_429(ctx: FaultContext) -> Response | Forward:
    return error_response(429, "rate_limit_error")


@faults("http_500")
async def http_500(ctx: FaultContext) -> Response | Forward:
    return error_response(500, "api_error")


@faults("latency")
async def latency(ctx: FaultContext) -> Response | Forward:
    await asyncio.sleep(ctx.latency_s)
    return Forward()


@faults("truncate")
async def truncate(ctx: FaultContext) -> Response | Forward:
    return Forward(max_chunks=1)


def check_rates(rates: Mapping[str, float]) -> None:
    for name, rate in rates.items():
        if name not in faults:
            raise ValueError(f"unknown fault {name!r} (known: {', '.join(faults.names())})")
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"fault rate for {name!r} must be between 0 and 1, got {rate}")


def pick_fault(rates: Mapping[str, float], *, seed: int, key: str, index: int) -> str | None:
    """The same (seed, fault key, call index) always gets the same fault, whatever the timing."""
    rng = random.Random(f"{seed}:{key}:{index}")
    for name in sorted(rates):
        if rng.random() < rates[name]:
            return name
    return None
