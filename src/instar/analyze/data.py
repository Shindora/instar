from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.events import (
    Execution,
    ExecutionStatus,
    LLMCall,
    Load,
    Metric,
    Run,
    Shed,
    StageWindow,
)
from instar.store import StoreReader

if TYPE_CHECKING:
    from pathlib import Path

type MetricValue = bool | int | float | str | None

COST_METRIC = "cost_usd"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunData:
    """Everything the analyzer reads, loaded once from a run store (finished or still running)."""

    run: Run | None
    executions: list[Execution]
    metrics: dict[str, dict[str, MetricValue]]
    calls: list[LLMCall]
    stages: list[StageWindow]
    loads: list[Load]
    sheds: list[Shed]

    @classmethod
    def load(cls, run_dir: Path) -> RunData:
        reader = StoreReader(run_dir)
        try:
            metrics: dict[str, dict[str, MetricValue]] = defaultdict(dict)
            for metric in reader.select(Metric):
                metrics[metric.session][metric.name] = metric.value
            runs = reader.select(Run)
            return cls(
                run=runs[-1] if runs else None,
                executions=reader.select(Execution),
                metrics=dict(metrics),
                calls=reader.select(LLMCall),
                stages=reader.select(StageWindow),
                loads=reader.select(Load),
                sheds=reader.select(Shed),
            )
        finally:
            reader.close()

    def finished(self) -> list[Execution]:
        return [e for e in self.executions if e.status is not ExecutionStatus.CANCELLED]

    @property
    def open_loop(self) -> bool:
        return self.run is not None and self.run.config.get("rate_per_s") is not None

    def cost(self, execution: Execution) -> float | None:
        value = self.metrics.get(execution.session, {}).get(COST_METRIC)
        return (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )
