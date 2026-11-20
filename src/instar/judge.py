from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from functools import singledispatch

type Metrics = dict[str, object]


class Direction(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricSpec:
    """How to read one metric: which way is better, and its worst value.

    The worst value is what a failed or unscored execution counts as when arms are compared.
    Higher-is-better metrics default to 0; a lower-is-better metric must declare it.
    """

    name: str
    direction: Direction = Direction.HIGHER
    worst: float | None = None

    @property
    def worst_value(self) -> float | None:
        if self.worst is not None:
            return self.worst
        return 0.0 if self.direction is Direction.HIGHER else None

    def better(self, difference: float) -> bool:
        return difference > 0 if self.direction is Direction.HIGHER else difference < 0


@singledispatch
def as_metrics(result: object) -> Metrics:
    """Turn a judge's return value (bool, number, mapping or dataclass) into a metric mapping."""
    if is_dataclass(result) and not isinstance(result, type):
        return as_metrics(asdict(result))
    raise TypeError(f"judge returned unsupported type {type(result).__name__}")


@as_metrics.register(bool)
def _(result: bool) -> Metrics:
    return {"passed": result}


@as_metrics.register(int)
@as_metrics.register(float)
def _(result: float) -> Metrics:
    return {"score": float(result)}


@as_metrics.register(Mapping)
def _(result: Mapping[object, object]) -> Metrics:
    return {str(k): v for k, v in result.items()}
