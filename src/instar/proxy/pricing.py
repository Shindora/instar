from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from instar.proxy.usage import Usage

PRICE_HINT = "add it under [tool.instar.prices]"
GLOB = frozenset("*?[")


@dataclass(frozen=True, slots=True, kw_only=True)
class Price:
    """USD per million tokens. A cache price left out falls back to `input`, and a cost that
    used such a fallback is marked approximate: cache prices differ by model."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    cache_write_1h: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    usd: float | None
    note: str | None = None
    approximate: bool = False


def find_price(prices: Mapping[str, Price], model: str) -> Price | None:
    """An exact key wins; then the longest glob key that matches. No implicit prefixes."""
    if model in prices:
        return prices[model]
    patterns = [key for key in prices if GLOB & set(key) and fnmatch.fnmatchcase(model, key)]
    return prices[max(patterns, key=len)] if patterns else None


def call_cost(prices: Mapping[str, Price], model: str | None, status: int, usage: Usage) -> Cost:
    """A call's cost in USD from the price table, or None with the reason."""
    if status >= 400:
        return Cost(usd=0.0)
    if model is None:
        return Cost(usd=None, note="the request named no model")
    price = find_price(prices, model)
    if price is None:
        return Cost(usd=None, note=f"no price for model {model!r}")
    if not usage.known:
        return Cost(usd=None, note="usage unknown: the response ended without token usage")
    approximate = False

    def rate(value: float | None, tokens: int) -> float:
        nonlocal approximate
        if value is None and tokens:
            approximate = True
        return price.input if value is None else value

    read, writes, one_hour = (
        usage.cached_input_tokens,
        usage.cache_write_tokens,
        usage.cache_write_1h_tokens,
    )
    fresh = max(usage.input_tokens - read - writes, 0)
    usd = (
        fresh * price.input
        + read * rate(price.cache_read, read)
        + (writes - one_hour) * rate(price.cache_write, writes - one_hour)
        + one_hour * rate(price.cache_write_1h, one_hour)
        + usage.output_tokens * price.output
    )
    return Cost(usd=usd / 1e6, approximate=approximate)
