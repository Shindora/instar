"""Paired comparison of two arms that differ by one environment variable."""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from instar.analyze.curve import is_bool_metric, primary_estimate
from instar.analyze.stats import RESAMPLES, Z_95, Estimate, percentile
from instar.events import ExecutionStatus
from instar.judge import Direction, MetricSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from instar.analyze.data import RunData
    from instar.events import Execution

ALPHA = 0.05
Z_POWER_80 = 0.841621
EXACT_DISCORDANT = 6  # fewest one-way discordant pairs for exact McNemar p < 0.05: 2 / 2**6
POWER_RULE = "80% power"
EXACT_RULE = "minimum that can reach p < 0.05"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArmStats:
    name: str
    executions: int
    failures: dict[str, int]
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
    mcnemar_p: float | None = None
    smallest_detectable: float | None = None
    smallest_detectable_rule: str | None = None

    @property
    def detected(self) -> bool:
        return not self.low <= 0 <= self.high


@dataclass(frozen=True, slots=True, kw_only=True)
class ABResult:
    variable: str
    a: ArmStats
    b: ArmStats
    direction: Direction
    worst: float
    primary: Difference | None
    cost: Difference | None
    run_s: Difference | None
    percent: bool

    @property
    def scoring(self) -> str:
        if self.percent:
            return "A failed or unjudged execution counts as not passed."
        return (
            f"{self.direction.capitalize()} is better; a failed or unscored execution counts as"
            f" the worst value, {self.worst:g}."
        )

    @property
    def failures(self) -> str:
        def counts(arm: ArmStats) -> str:
            listed = ", ".join(f"{n} {status}" for status, n in sorted(arm.failures.items()))
            return f"{arm.name} {listed or 'none'}"

        return f"Failures: {counts(self.a)}; {counts(self.b)}."

    def sentence(self) -> str:
        diff = self.primary
        if diff is None:
            return f"No complete pairs of {self.a.name} and {self.b.name} to compare."
        show = _percent if self.percent else _number
        span = f"{show(diff.low, True)} to {show(diff.high, True)}"
        counted = f"over {diff.pairs} pairs in {diff.tasks} tasks"
        if diff.detected:
            spec = MetricSpec(name=diff.metric, direction=self.direction)
            verdict = "better" if spec.better(diff.value) else "worse"
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
                    f" {show(diff.smallest_detectable, False)} or more"
                    f" ({diff.smallest_detectable_rule})."
                )
        if diff.mcnemar_p is not None:
            text += f" Exact McNemar p = {diff.mcnemar_p:.3g}."
        return f"{text} {self.failures}"


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


type Pairs = dict[str, list[tuple[float, float]]]


def _difference(metric: str, pairs: Pairs, *, binary: bool, seed: str) -> Difference:
    per_task = [statistics.fmean(b - a for a, b in task_pairs) for task_pairs in pairs.values()]
    low, high = _bootstrap_tasks(per_task, seed)
    mcnemar = None
    if binary and all(len(p) == 1 for p in pairs.values()):
        b_only = sum(1 for [(a, b)] in pairs.values() if b > a)
        a_only = sum(1 for [(a, b)] in pairs.values() if a > b)
        mcnemar = mcnemar_exact(b_only, a_only)
    standard_error = (high - low) / (2 * Z_95)
    smallest, rule = None, None
    if standard_error > 0:
        smallest, rule = (Z_95 + Z_POWER_80) * standard_error, POWER_RULE
    elif binary:
        smallest, rule = EXACT_DISCORDANT / len(per_task), EXACT_RULE
    return Difference(
        metric=metric,
        value=statistics.fmean(per_task),
        low=low,
        high=high,
        pairs=sum(len(p) for p in pairs.values()),
        tasks=len(per_task),
        mcnemar_p=mcnemar,
        smallest_detectable=smallest,
        smallest_detectable_rule=rule,
    )


def _arm(
    data: RunData, name: str, executions: list[Execution], primary: str, seed: str
) -> ArmStats:
    sessions = {e.session for e in executions}
    calls = [c for c in data.calls if c.session in sessions and c.role == "agent"]
    tokens = sum(c.input_tokens for c in calls)
    costs = [data.cost(e) for e in executions]
    passes = sum(1 for e in executions if data.metrics.get(e.session, {}).get(primary) is True)
    known = [c for c in costs if c is not None]
    return ArmStats(
        name=name,
        executions=len(executions),
        failures=dict(
            Counter(str(e.status) for e in executions if e.status is not ExecutionStatus.OK)
        ),
        primary=primary_estimate(data, executions, primary, seed=seed),
        cost_per_pass=sum(known) / passes if passes and len(known) == len(costs) else None,
        p50_run_s=percentile([e.run_ns / 1e9 for e in executions], 0.5),
        cached_share=sum(c.cached_input_tokens for c in calls) / tokens if tokens else None,
    )


def _secondary(
    complete: Sequence[tuple[str, dict[str, Execution]]],
    a: str,
    b: str,
    value: Callable[[Execution], float | None],
) -> Pairs:
    """Pairs where both arms finished and the value exists on both sides; nothing imputed."""
    pairs: Pairs = defaultdict(list)
    for task, arms in complete:
        if arms[a].status is not ExecutionStatus.OK or arms[b].status is not ExecutionStatus.OK:
            continue
        first, second = value(arms[a]), value(arms[b])
        if first is not None and second is not None:
            pairs[task].append((first, second))
    return pairs


def compare_arms(
    data: RunData,
    *,
    variable: str,
    a: str,
    b: str,
    spec: MetricSpec,
    seed: str,
) -> ABResult:
    """Primary: every pair, with failures scored as the worst value. Cost and run time: only
    pairs where both arms finished."""
    primary = spec.name
    finished = data.finished()
    by_key: dict[tuple[int, str, int], dict[str, Execution]] = defaultdict(dict)
    for execution in finished:
        by_key[execution.stage, execution.task_id, execution.repeat][execution.arm] = execution
    complete = [(key[1], arms) for key, arms in by_key.items() if a in arms and b in arms]
    binary = is_bool_metric(data, primary)
    worst = 0.0 if binary else spec.worst_value
    if worst is None:
        raise ValueError(f"metric {primary!r} is lower-is-better and declares no worst value")

    def score(execution: Execution) -> float:
        value = data.metrics.get(execution.session, {}).get(primary)
        if binary:
            return 1.0 if value is True else 0.0
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        return worst

    scores: Pairs = defaultdict(list)
    for task, arms in complete:
        scores[task].append((score(arms[a]), score(arms[b])))
    costs = _secondary(complete, a, b, data.cost)
    run_s = _secondary(complete, a, b, lambda e: e.run_ns / 1e9)
    in_arm = {name: [e for e in finished if e.arm == name] for name in (a, b)}
    return ABResult(
        variable=variable,
        a=_arm(data, a, in_arm[a], primary, f"{seed}:arm-a"),
        b=_arm(data, b, in_arm[b], primary, f"{seed}:arm-b"),
        direction=Direction.HIGHER if binary else spec.direction,
        worst=worst,
        primary=_difference(primary, scores, binary=binary, seed=f"{seed}:ab") if scores else None,
        cost=_difference("cost_usd", costs, binary=False, seed=f"{seed}:cost") if costs else None,
        run_s=_difference("run_s", run_s, binary=False, seed=f"{seed}:run") if run_s else None,
        percent=binary,
    )
