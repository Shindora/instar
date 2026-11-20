from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

PRICE_HINT = "add it under [tool.instar.prices]"


@dataclass(frozen=True, slots=True, kw_only=True)
class Price:
    """USD per million tokens. Cached input tokens cost `cached`, or `input` when it is unset."""

    input: float
    output: float
    cached: float | None = None


def call_cost(
    prices: Mapping[str, Price],
    model: str | None,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> tuple[float | None, str | None]:
    """The call's cost in USD, or None and the reason it is unknown."""
    if model is None:
        return None, "the request named no model"
    price = prices.get(model)
    if price is None:
        return None, f"no price for model {model!r}"
    cached_rate = price.input if price.cached is None else price.cached
    fresh = max(input_tokens - cached_input_tokens, 0)
    usd = fresh * price.input + cached_input_tokens * cached_rate + output_tokens * price.output
    return usd / 1e6, None
