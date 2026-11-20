from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from functools import singledispatch

type Metrics = dict[str, object]


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
