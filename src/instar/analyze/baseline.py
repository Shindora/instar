from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from instar.config import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from instar.analyze import Analysis


@dataclass(frozen=True, slots=True, kw_only=True)
class Baseline:
    run_id: str
    date: str
    seed: int
    models: list[str]
    evaluation_version: str | None
    primary: str
    summary: dict[str, float | None]
    tasks: dict[str, dict[str, object]]

    def primary_over(self, task_ids: Sequence[str]) -> float | None:
        """The baseline's primary metric, averaged over the given tasks only."""
        values = [v for t in task_ids if isinstance(v := self.tasks[t].get("primary"), int | float)]
        return statistics.fmean(values) if values else None


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskChange:
    """The primary metric of one task present in both runs, in the baseline and now."""

    task_id: str
    before: float | None
    after: float | None

    @property
    def regressed(self) -> bool:
        return self.before is not None and (self.after is None or self.after < self.before)

    @property
    def improved(self) -> bool:
        return self.before is not None and self.after is not None and self.after > self.before


@dataclass(frozen=True, slots=True, kw_only=True)
class Comparison:
    baseline_run: str
    changes: list[TaskChange]
    only_now: list[str] = field(default_factory=list)
    only_baseline: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def matched(self) -> list[str]:
        return [change.task_id for change in self.changes]

    @property
    def regressed(self) -> list[TaskChange]:
        return [change for change in self.changes if change.regressed]


def document(analysis: Analysis) -> dict[str, object]:
    return {
        "run_id": analysis.run_id,
        "date": analysis.started_at,
        "seed": analysis.seed,
        "models": analysis.models,
        "evaluation_version": analysis.evaluation_version,
        "project": analysis.project,
        "primary": analysis.primary,
        "task_ids": sorted(row.task_id for row in analysis.tasks),
        "summary": {name: estimate.value for name, estimate in analysis.metrics.items()},
        "tasks": {
            row.task_id: {
                "primary": row.primary,
                "executions": row.executions,
                "cost_usd": row.cost_usd,
                "metrics": row.metrics,
            }
            for row in analysis.tasks
        },
    }


def save(analysis: Analysis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document(analysis), indent=2) + "\n")


def load(path: Path) -> Baseline:
    try:
        data = json.loads(path.read_text())
        return Baseline(
            run_id=data["run_id"],
            date=data["date"],
            seed=data["seed"],
            models=data["models"],
            evaluation_version=data["evaluation_version"],
            primary=data["primary"],
            summary=data["summary"],
            tasks=data["tasks"],
        )
    except FileNotFoundError:
        raise ConfigError(f"baseline not found: {path}") from None
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"baseline {path} is not an instar baseline: {exc!r}") from None


def compare(analysis: Analysis, baseline: Baseline) -> Comparison:
    """Per-task changes over the tasks both runs share, regressions first."""
    now = {row.task_id: row.primary for row in analysis.tasks}
    matched = sorted(now.keys() & baseline.tasks.keys())
    changes = []
    for task in matched:
        before = baseline.tasks[task].get("primary")
        changes.append(
            TaskChange(
                task_id=task,
                before=float(before) if isinstance(before, int | float) else None,
                after=now[task],
            )
        )
    warnings = []
    if sorted(baseline.models) != sorted(analysis.models):
        warnings.append(
            f"model ids differ: baseline {', '.join(baseline.models) or 'none'},"
            f" now {', '.join(analysis.models) or 'none'}"
        )
    if baseline.evaluation_version != analysis.evaluation_version:
        warnings.append(
            f"evaluation code differs: baseline {baseline.evaluation_version},"
            f" now {analysis.evaluation_version}"
        )
    if baseline.primary != analysis.primary:
        warnings.append(
            f"primary metric differs: baseline {baseline.primary}, now {analysis.primary}"
        )
    return Comparison(
        baseline_run=baseline.run_id,
        changes=sorted(changes, key=lambda c: (not c.regressed, c.task_id)),
        only_now=sorted(now.keys() - baseline.tasks.keys()),
        only_baseline=sorted(baseline.tasks.keys() - now.keys()),
        warnings=warnings,
    )


def iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat(timespec="seconds")
