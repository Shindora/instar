from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.config import ConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from instar.analyze import Analysis, TaskRow


@dataclass(frozen=True, slots=True, kw_only=True)
class Baseline:
    run_id: str
    primary: str
    summary: dict[str, float | None]
    tasks: dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskChange:
    """The primary metric of one task, in the baseline and now."""

    task_id: str
    before: float | None
    after: float | None

    @property
    def regressed(self) -> bool:
        return self.before is not None and (self.after is None or self.after < self.before)

    @property
    def improved(self) -> bool:
        return self.before is not None and self.after is not None and self.after > self.before


def document(analysis: Analysis) -> dict[str, object]:
    return {
        "run_id": analysis.run_id,
        "project": analysis.project,
        "primary": analysis.primary,
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
            primary=data["primary"],
            summary=data["summary"],
            tasks=data["tasks"],
        )
    except FileNotFoundError:
        raise ConfigError(f"baseline not found: {path}") from None
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"baseline {path} is not an instar baseline: {exc!r}") from None


def compare(rows: Sequence[TaskRow], baseline: Baseline) -> list[TaskChange]:
    """Every task present in the baseline or now, regressions first."""
    now = {row.task_id: row.primary for row in rows}
    before = {task: values.get("primary") for task, values in baseline.tasks.items()}
    changes = [
        TaskChange(
            task_id=task,
            before=float(b) if isinstance(b := before.get(task), int | float) else None,
            after=now.get(task),
        )
        for task in sorted(before.keys() | now.keys())
    ]
    return sorted(changes, key=lambda c: (not c.regressed, c.task_id))
