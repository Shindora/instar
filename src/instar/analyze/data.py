from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
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


class CostStatus(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    LOWER_BOUND = "lower bound"
    INCOMPLETE = "incomplete"


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

    def _agent_calls(self) -> list[LLMCall]:
        return [c for c in self.calls if c.role == "agent"]

    def unpriced_models(self) -> list[str]:
        """Models of agent calls the price table does not cover."""
        unpriced = [c for c in self._agent_calls() if c.cost_usd is None and c.usage_known]
        return sorted({c.model or "<no model in request>" for c in unpriced})

    def usage_unknown(self) -> int:
        """Agent calls whose response ended without token usage, so their cost is unknown."""
        return sum(1 for c in self._agent_calls() if not c.usage_known)

    def cost_status(self) -> CostStatus:
        if self.unpriced_models():
            return CostStatus.INCOMPLETE
        if self.usage_unknown():
            return CostStatus.LOWER_BOUND
        if any(c.cost_approximate for c in self._agent_calls()):
            return CostStatus.APPROXIMATE
        return CostStatus.EXACT

    def cost(self, execution: Execution) -> float | None:
        """USD from the price table. None when a call has no price; a call that ended without
        usage adds nothing, so the sum is then a lower bound (see `cost_status`)."""
        calls = self.agent_calls(execution.session)
        if any(c.cost_usd is None and c.usage_known for c in calls):
            return None
        return sum(c.cost_usd or 0.0 for c in calls)

    def cost_note(self) -> str | None:
        """Why cost is incomplete, a lower bound or approximate."""
        status = self.cost_status()
        if status is CostStatus.INCOMPLETE:
            return f"no price for {', '.join(self.unpriced_models())} ({PRICE_HINT})"
        if status is CostStatus.LOWER_BOUND:
            return f"{self.usage_unknown()} model calls ended without token usage"
        if status is CostStatus.APPROXIMATE:
            return "a cache price is missing, so those tokens were priced as input"
        return None
