"""Paired comparison of two arms that differ by one environment variable."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.analyze.curve import is_bool_metric, primary_estimate
from instar.analyze.stats import RESAMPLES, Z_95, Estimate, percentile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from instar.analyze.data import RunData
    from instar.events import Execution

ALPHA = 0.05
Z_POWER_80 = 0.841621
EXACT_DISCORDANT = 6  # fewest one-sided discordant pairs for exact McNemar p < 0.05: 2 / 2**6


@dataclass(frozen=True, slots=True, kw_only=True)
class ArmStats:
    name: str
    executions: int
    primary: Estimate
    cost_per_pass: float | None
    p50_run_s: float | None
    cached_share: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Difference:
    """`b` minus `a`, averaged per task, with a bootstrap interval over tasks."""

    metric: str
    value: float
    low: float
    high: float
    pairs: int
    tasks: int
    mcnemar_p: float | None
    smallest_detectable: float | None

    @property
    def detected(self) -> bool:
        return not self.low <= 0 <= self.high


@dataclass(frozen=True, slots=True, kw_only=True)
class ABResult:
    variable: str
    a: ArmStats
    b: ArmStats
    primary: Difference | None
    cost: Difference | None
    percent: bool

    def sentence(self) -> str:
        diff = self.primary
        if diff is None:
            return f"No complete pairs of {self.a.name} and {self.b.name} to compare."
        show = _percent if self.percent else _number
        span = f"{show(diff.low, True)} to {show(diff.high, True)}"
        counted = f"over {diff.pairs} pairs in {diff.tasks} tasks"
        if diff.detected:
            verdict = "better" if diff.value > 0 else "worse"
            text = (
                f"{self.variable}={self.b.name} does {verdict} than {self.a.name}:"
                f" {diff.metric} {show(diff.value, True)} ({span}) {counted}."
            )
        else:
            text = (
                f"No difference detected between {self.a.name} and {self.b.name}:"
                f" {diff.metric} {show(diff.value, True)} ({span}) {counted}."
            )
            if diff.smallest_detectable is not None:
                text += (
                    " This run could detect a difference of about"
                    f" {show(diff.smallest_detectable, False)} or more."
                )
        if diff.mcnemar_p is not None:
            text += f" Exact McNemar p = {diff.mcnemar_p:.3g}."
        return text


def _percent(value: float, signed: bool) -> str:
    return f"{value:+.0%}" if signed else f"{value:.0%}"


def _number(value: float, signed: bool) -> str:
    return f"{value:+.3g}" if signed else f"{value:.3g}"


def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b_only, a_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _bootstrap_tasks(per_task: Sequence[float], seed: str) -> tuple[float, float]:
    """Resample tasks, never executions: executions of one task are not independent."""
    if len(per_task) < 2:
        return per_task[0], per_task[0]
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(per_task, k=len(per_task))) for _ in range(RESAMPLES)]
    low, high = percentile(means, ALPHA / 2), percentile(means, 1 - ALPHA / 2)
    assert low is not None and high is not None
    return low, high


def _difference(
    metric: str,
    pairs: dict[str, list[tuple[float, float]]],
    *,
    binary: bool,
    seed: str,
) -> Difference:
    per_task = [statistics.fmean(b - a for a, b in task_pairs) for task_pairs in pairs.values()]
    value = statistics.fmean(per_task)
    low, high = _bootstrap_tasks(per_task, seed)
    count = sum(len(p) for p in pairs.values())
    one_pair_each = all(len(p) == 1 for p in pairs.values())
    mcnemar = None
    if binary and one_pair_each:
        b_only = sum(1 for [(a, b)] in pairs.values() if b > a)
        a_only = sum(1 for [(a, b)] in pairs.values() if a > b)
        mcnemar = mcnemar_exact(b_only, a_only)
    standard_error = (high - low) / (2 * Z_95)
    if standard_error > 0:
        smallest: float | None = (Z_95 + Z_POWER_80) * standard_error
    elif binary:
        smallest = EXACT_DISCORDANT / len(per_task)
    else:
        smallest = None
    return Difference(
        metric=metric,
        value=value,
        low=low,
        high=high,
        pairs=count,
        tasks=len(per_task),
        mcnemar_p=mcnemar,
        smallest_detectable=smallest,
    )


def _arm(
    data: RunData, name: str, executions: list[Execution], primary: str, seed: str
) -> ArmStats:
    sessions = {e.session for e in executions}
    calls = [c for c in data.calls if c.session in sessions and c.role == "agent"]
    tokens = sum(c.input_tokens for c in calls)
    costs = [c for e in executions if (c := data.cost(e)) is not None]
    passes = sum(1 for e in executions if data.metrics.get(e.session, {}).get(primary) is True)
    return ArmStats(
        name=name,
        executions=len(executions),
        primary=primary_estimate(data, executions, primary, seed=seed),
        cost_per_pass=sum(costs) / passes if costs and passes else None,
        p50_run_s=percentile([e.run_ns / 1e9 for e in executions], 0.5),
        cached_share=sum(c.cached_input_tokens for c in calls) / tokens if tokens else None,
    )


def _score(data: RunData, execution: Execution, primary: str, binary: bool) -> float:
    """Timeouts, errors and crashes score as failures, never as missing."""
    value = data.metrics.get(execution.session, {}).get(primary)
    if binary:
        return 1.0 if value is True else 0.0
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def compare_arms(
    data: RunData, *, variable: str, a: str, b: str, primary: str, seed: str
) -> ABResult:
    finished = data.finished()
    by_key: dict[tuple[int, str, int], dict[str, Execution]] = defaultdict(dict)
    for execution in finished:
        by_key[execution.stage, execution.task_id, execution.repeat][execution.arm] = execution
    complete = [(key, arms) for key, arms in by_key.items() if a in arms and b in arms]
    binary = is_bool_metric(data, primary)
    scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    costs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (_, task, _), arms in complete:
        scores[task].append(
            (_score(data, arms[a], primary, binary), _score(data, arms[b], primary, binary))
        )
        cost_a, cost_b = data.cost(arms[a]), data.cost(arms[b])
        if cost_a is not None and cost_b is not None:
            costs[task].append((cost_a, cost_b))
    in_arm = {name: [e for e in finished if e.arm == name] for name in (a, b)}
    return ABResult(
        variable=variable,
        a=_arm(data, a, in_arm[a], primary, f"{seed}:arm-a"),
        b=_arm(data, b, in_arm[b], primary, f"{seed}:arm-b"),
        primary=_difference(primary, scores, binary=binary, seed=f"{seed}:ab") if scores else None,
        cost=_difference("cost_usd", costs, binary=False, seed=f"{seed}:ab-cost")
        if costs
        else None,
        percent=binary,
    )
