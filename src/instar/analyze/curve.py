from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.analyze.data import COST_METRIC
from instar.analyze.stats import Estimate, mean, percentile, rate
from instar.events import ExecutionStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from instar.analyze.data import RunData
    from instar.events import Execution

FAILED_STATUSES = frozenset({401, 403, 407, 502})


@dataclass(frozen=True, slots=True, kw_only=True)
class Group:
    """Statistics for any set of executions: a stage, first attempts, one task."""

    executions: int
    statuses: dict[str, int]
    primary: Estimate
    p50_run_s: float | None
    p95_run_s: float | None
    cost_usd: float | None
    cost_per_pass: float | None
    model_calls: int
    model_errors: int
    faults: int

    @property
    def model_error_rate(self) -> float | None:
        return self.model_errors / self.model_calls if self.model_calls else None


@dataclass(frozen=True, slots=True, kw_only=True)
class StageStats:
    stage: int
    users: int
    hold_s: float | None
    steady: Group
    ramping: int
    tasks_per_hour: float | None
    spend_per_hour: float | None
    arrivals: int | None
    shed: int

    @property
    def shed_rate(self) -> float | None:
        return self.shed / self.arrivals if self.arrivals else None


@dataclass(frozen=True, slots=True, kw_only=True)
class BreakingPoint:
    stage: int
    users: int
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Bucket:
    """One interval of the time series; `users` is the most users planned during it."""

    start_s: float
    users: int
    started: int
    finished: int
    pass_rate: float | None
    p95_run_s: float | None
    model_errors: int
    spend_usd: float | None


def is_bool_metric(data: RunData, name: str) -> bool:
    values = [m[name] for m in data.metrics.values() if name in m]
    return bool(values) and all(isinstance(v, bool) for v in values)


def primary_estimate(
    data: RunData, executions: Sequence[Execution], name: str, *, seed: str
) -> Estimate:
    """Boolean primary: a rate over every finished execution, where no verdict means not passed.

    Numeric primary: the mean over executions that reported it.
    """
    values = [data.metrics.get(e.session, {}).get(name) for e in executions]
    if is_bool_metric(data, name):
        return rate([v is True for v in values])
    numbers = [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
    return mean(numbers, seed=seed)


def _passes(data: RunData, executions: Sequence[Execution], name: str) -> int:
    return sum(1 for e in executions if data.metrics.get(e.session, {}).get(name) is True)


def group(data: RunData, executions: Sequence[Execution], name: str, *, seed: str) -> Group:
    sessions = {e.session for e in executions}
    calls = [c for c in data.calls if c.session in sessions and c.role == "agent"]
    run_s = [e.run_ns / 1e9 for e in executions]
    costs = [c for e in executions if (c := data.cost(e)) is not None]
    cost = sum(costs) if costs else None
    passes = _passes(data, executions, name)
    return Group(
        executions=len(executions),
        statuses=dict(Counter(str(e.status) for e in executions)),
        primary=primary_estimate(data, executions, name, seed=seed),
        p50_run_s=percentile(run_s, 0.5),
        p95_run_s=percentile(run_s, 0.95),
        cost_usd=cost,
        cost_per_pass=cost / passes if cost is not None and passes else None,
        model_calls=len(calls),
        model_errors=sum(1 for c in calls if c.status >= 400),
        faults=sum(1 for c in calls if c.fault is not None),
    )


def stage_stats(data: RunData, name: str, *, seed: str) -> list[StageStats]:
    finished = data.finished()
    result = []
    for window in sorted(data.stages, key=lambda w: w.stage):
        in_stage = [e for e in finished if e.stage == window.stage]
        steady = [e for e in in_stage if not e.ramping]
        hours = window.hold_s / 3600 if window.hold_s else None
        stats = group(data, steady, name, seed=f"{seed}:stage{window.stage}")
        shed = sum(1 for s in data.sheds if s.stage == window.stage)
        result.append(
            StageStats(
                stage=window.stage,
                users=window.users,
                hold_s=window.hold_s,
                steady=stats,
                ramping=len(in_stage) - len(steady),
                tasks_per_hour=len(steady) / hours if hours else None,
                spend_per_hour=stats.cost_usd / hours
                if hours and stats.cost_usd is not None
                else None,
                arrivals=len(in_stage) + shed if data.open_loop else None,
                shed=shed,
            )
        )
    return result


def breaking_point(stages: Sequence[StageStats], p95_multiple: float) -> BreakingPoint | None:
    """The first stage whose primary falls below the first stage's interval, or whose p95 run
    time exceeds `p95_multiple` times the first stage's."""
    if len(stages) < 2:
        return None
    base = stages[0].steady
    for stage in stages[1:]:
        current = stage.steady
        low, value = base.primary.low, current.primary.value
        if low is not None and value is not None and value < low:
            return BreakingPoint(
                stage=stage.stage,
                users=stage.users,
                reason=(
                    f"primary {value:.0%} fell below stage {stages[0].stage}'s interval"
                    f" ({low:.0%}-{base.primary.high or 1:.0%})"
                ),
            )
        if (
            base.p95_run_s
            and current.p95_run_s
            and current.p95_run_s > p95_multiple * base.p95_run_s
        ):
            return BreakingPoint(
                stage=stage.stage,
                users=stage.users,
                reason=(
                    f"p95 run time {current.p95_run_s:.2f} s exceeds {p95_multiple:g}x stage"
                    f" {stages[0].stage}'s {base.p95_run_s:.2f} s"
                ),
            )
    return None


def time_series(data: RunData, name: str, interval_s: float) -> list[Bucket]:
    starts = [e.started_ns for e in data.executions] + [load.at_ns for load in data.loads]
    if not starts:
        return []
    origin = min(starts)
    ends = [e.ended_ns for e in data.executions] + starts
    width = int(interval_s * 1e9)
    count = (max(ends) - origin) // width + 1
    loads = sorted(data.loads, key=lambda load: load.at_ns)
    buckets = []
    for i in range(count):
        low, high = origin + i * width, origin + (i + 1) * width
        before = [load.users for load in loads if load.at_ns <= low]
        during = [load.users for load in loads if low < load.at_ns < high]
        planned = before[-1:] + during
        finished = [e for e in data.finished() if low <= e.ended_ns < high]
        flags = [data.metrics.get(e.session, {}).get(name) is True for e in finished]
        costs = [c for e in finished if (c := data.cost(e)) is not None]
        buckets.append(
            Bucket(
                start_s=i * interval_s,
                users=max(planned, default=0),
                started=sum(1 for e in data.executions if low <= e.started_ns < high),
                finished=len(finished),
                pass_rate=sum(flags) / len(flags) if flags and is_bool_metric(data, name) else None,
                p95_run_s=percentile([e.run_ns / 1e9 for e in finished], 0.95),
                model_errors=sum(
                    1 for c in data.calls if low <= c.started_ns < high and c.status >= 400
                ),
                spend_usd=sum(costs) if costs else None,
            )
        )
    return buckets


def metric_names(data: RunData) -> list[str]:
    return sorted({n for m in data.metrics.values() for n in m} - {COST_METRIC})


def could_not_run(data: RunData) -> str | None:
    """Why the test itself could not run (exit code 3), as opposed to the agent doing badly."""
    finished = data.finished()
    if not finished or any(e.status is ExecutionStatus.OK for e in finished):
        return None
    real = [c for c in data.calls if c.fault is None]
    if real and all(c.status in FAILED_STATUSES for c in real):
        codes = ", ".join(
            f"HTTP {s} x{n}" for s, n in sorted(Counter(c.status for c in real).items())
        )
        return f"every model call failed ({codes}): check credentials and that the upstream is up"
    return None
