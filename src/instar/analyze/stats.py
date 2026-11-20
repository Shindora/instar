from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Z_95 = 1.959964
RESAMPLES = 2000


@dataclass(frozen=True, slots=True, kw_only=True)
class Estimate:
    """A point estimate with its 95% interval; `low`/`high` are None when there is no interval."""

    value: float | None
    low: float | None = None
    high: float | None = None
    n: int = 0

    def contains(self, threshold: float) -> bool:
        if self.low is None or self.high is None:
            return False
        return self.low <= threshold <= self.high


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear interpolation between closest ranks, as numpy's default."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a rate; well-behaved at 0% and 100% and for small n."""
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(
    values: Sequence[float], *, seed: str, resamples: int = RESAMPLES
) -> tuple[float, float] | None:
    """Seeded percentile bootstrap interval of the mean."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples)]
    low, high = percentile(means, 0.025), percentile(means, 0.975)
    assert low is not None and high is not None
    return low, high


def rate(flags: Sequence[bool]) -> Estimate:
    passed = sum(flags)
    interval = wilson(passed, len(flags))
    if interval is None:
        return Estimate(value=None)
    return Estimate(value=passed / len(flags), low=interval[0], high=interval[1], n=len(flags))


def mean(values: Sequence[float], *, seed: str) -> Estimate:
    if not values:
        return Estimate(value=None)
    interval = bootstrap_mean(values, seed=seed)
    low, high = interval if interval else (None, None)
    return Estimate(value=statistics.fmean(values), low=low, high=high, n=len(values))
