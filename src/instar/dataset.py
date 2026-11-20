from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True, slots=True, kw_only=True)
class Task:
    """One unit of work: an id plus whatever the agent and judge need, read as attributes.

    `task.prompt` reads `task.data["prompt"]`. Everything must be JSON-serialisable, because
    a task travels to its worker process as JSON.
    """

    id: str
    data: Mapping[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> object:
        if name == "data":
            raise AttributeError(name)
        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(f"task {self.id!r} has no field {name!r}") from None


def _tasks(rows: Iterable[tuple[str, Mapping[str, object]]], id_field: str) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    for where, row in rows:
        if row.get(id_field) in (None, ""):
            raise ValueError(f"{where}: no {id_field!r} field")
        task_id = str(row[id_field])
        if task_id in seen:
            raise ValueError(f"{where}: duplicate task id {task_id!r}")
        seen.add(task_id)
        tasks.append(Task(id=task_id, data=dict(row)))
    return tasks


def from_jsonl(path: str | Path, *, id_field: str = "id") -> list[Task]:
    """One task per JSON object per line; blank lines are skipped."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append((f"{path}:{number}", row))
    return _tasks(rows, id_field)


def from_csv(path: str | Path, *, id_field: str = "id") -> list[Task]:
    """One task per row; every value is a string."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as file:
        rows = [(f"{path}:{n}", row) for n, row in enumerate(csv.DictReader(file), start=2)]
    return _tasks(rows, id_field)


def limit(tasks: Iterable[Task], n: int | None) -> list[Task]:
    """The first `n` tasks, or all of them when `n` is None."""
    tasks = list(tasks)
    return tasks if n is None else tasks[:n]
