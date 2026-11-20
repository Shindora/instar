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
from instar.proxy.pricing import PRICE_HINT
from instar.store import StoreReader

if TYPE_CHECKING:
    from pathlib import Path

type MetricValue = bool | int | float | str | None


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

    def agent_calls(self, session: str) -> list[LLMCall]:
        return [c for c in self.calls if c.session == session and c.role == "agent"]

    def cost(self, execution: Execution) -> float | None:
        """USD from the proxy's price table; None when any of its model calls has no price."""
        costs = [c.cost_usd for c in self.agent_calls(execution.session)]
        return None if any(c is None for c in costs) else sum(c or 0.0 for c in costs)

    def cost_note(self) -> str | None:
        """Why cost is unknown, naming the models with no price."""
        unknown = [c for c in self.calls if c.role == "agent" and c.cost_usd is None]
        notes = sorted({c.cost_note or "cost unknown" for c in unknown})
        return f"{'; '.join(notes)} ({PRICE_HINT})" if notes else None
