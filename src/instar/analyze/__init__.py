"""Turn a run store into numbers: one `analyze()` used by reports, gates, and the live UI."""

from __future__ import annotations

import copy
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from instar.analyze import baseline as baselines
from instar.analyze.ab import ABResult, compare_arms
from instar.analyze.curve import (
    BreakingPoint,
    Bucket,
    Group,
    StageStats,
    breaking_point,
    could_not_run,
    group,
    is_bool_metric,
    metric_names,
    primary_estimate,
    stage_stats,
    time_series,
)
from instar.analyze.data import CostStatus, RunData
from instar.analyze.gate import GateResult, evaluate, parse_all
from instar.analyze.stats import Estimate, mean, rate
from instar.config import ConfigError
from instar.events import ExecutionStatus
from instar.judge import Direction, MetricSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from instar.analyze.baseline import Baseline, Comparison
    from instar.analyze.data import MetricValue
    from instar.events import Execution
    from instar.runner import RunOptions


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRow:
    task_id: str
    executions: int
    statuses: dict[str, int]
    primary: float | None
    cost_usd: float | None
    metrics: dict[str, float]

    @property
    def failing(self) -> bool:
        low_primary = self.primary is not None and self.primary < 1
        return low_primary or any(status != "ok" for status in self.statuses)


@dataclass(frozen=True, slots=True, kw_only=True)
class Analysis:
    run_id: str
    project: str
    primary: str
    summary: Group
    first_attempts: Group
    repeats: Group | None
    stages: list[StageStats]
    breaking_point: BreakingPoint | None
    series: list[Bucket]
    tasks: list[TaskRow]
    metrics: dict[str, Estimate]
    percent_metrics: frozenset[str]
    started_at: str
    seed: int
    models: list[str]
    evaluation_version: str | None
    gates: list[GateResult] = field(default_factory=list)
    comparison: Comparison | None = None
    ab: ABResult | None = None
    could_not_run: str | None = None
    gate_error: str | None = None
    cost_note: str | None = None
    cost_status: str = "exact"
    unpriced_models: list[str] = field(default_factory=list)
    usage_unknown: int = 0
    token_cap: dict[str, int] | None = None
    notes: dict[str, str] = field(default_factory=dict)


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, int | float) else None


def _numbers(values: Sequence[Mapping[str, MetricValue]], name: str) -> list[float]:
    return [n for v in values if (n := _numeric(v.get(name))) is not None]


def _task_rows(data: RunData, executions: list[Execution], primary: str) -> list[TaskRow]:
    by_task: dict[str, list[Execution]] = {}
    for execution in executions:
        by_task.setdefault(execution.task_id, []).append(execution)
    bool_primary = is_bool_metric(data, primary)
    rows = []
    for task_id, runs in by_task.items():
        values = [data.metrics.get(e.session, {}) for e in runs]
        if bool_primary:
            primary_value: float | None = sum(v.get(primary) is True for v in values) / len(runs)
        else:
            numbers = _numbers(values, primary)
            primary_value = statistics.fmean(numbers) if numbers else None
        names = sorted({name for v in values for name in v})
        means = {name: statistics.fmean(ns) for name in names if (ns := _numbers(values, name))}
        costs = [c for e in runs if (c := data.cost(e)) is not None]
        rows.append(
            TaskRow(
                task_id=task_id,
                executions=len(runs),
                statuses=dict(Counter(str(e.status) for e in runs)),
                primary=primary_value,
                cost_usd=statistics.fmean(costs) if costs else None,
                metrics=means,
            )
        )
    return sorted(rows, key=lambda r: (not r.failing, r.primary or 0, r.task_id))


def _gate_metrics(
    data: RunData, steady: list[Execution], summary: Group, stages: list[StageStats], seed: str
) -> dict[str, Estimate]:
    metrics: dict[str, Estimate] = {}
    values = [data.metrics.get(e.session, {}) for e in steady]
    for name in metric_names(data):
        if is_bool_metric(data, name):
            metrics[name] = rate([v.get(name) is True for v in values])
        else:
            metrics[name] = mean(_numbers(values, name), seed=f"{seed}:{name}")
    metrics["timeout_rate"] = rate([e.status is ExecutionStatus.TIMEOUT for e in steady])
    metrics["error_rate"] = Estimate(value=summary.model_error_rate, n=summary.model_calls)
    lower_bound = data.cost_status() is CostStatus.LOWER_BOUND
    metrics["cost_per_pass"] = Estimate(value=summary.cost_per_pass, lower_bound=lower_bound)
    metrics["p50_run_s"] = Estimate(value=summary.p50_run_s, n=summary.executions)
    metrics["p95_run_s"] = Estimate(value=summary.p95_run_s, n=summary.executions)
    metrics["tasks_per_hour"] = Estimate(value=None)
    metrics["shed_rate"] = Estimate(value=None)
    timed = [s for s in stages if s.hold_s is not None]
    if timed:
        peak = max(reversed(timed), key=lambda s: s.users)
        metrics["tasks_per_hour"] = Estimate(value=peak.tasks_per_hour)
        if peak.arrivals is not None:
            metrics["shed_rate"] = Estimate(value=peak.shed_rate, n=peak.arrivals)
    return metrics


def _token_cap(config: dict[str, object]) -> dict[str, int] | None:
    """{"cap": N, "used": M} when the run hit its token cap."""
    reached = config.get("token_cap_reached")
    if not isinstance(reached, dict):
        return None
    return {str(k): int(v) for k, v in reached.items()}


def _notes(data: RunData, summary: Group, metrics: dict[str, Estimate]) -> dict[str, str]:
    """Why each metric without a value has none."""
    cost = data.cost_note()
    reasons = {
        "cost_per_pass": f"total cost is incomplete: {cost}" if cost else "no execution passed",
        "tasks_per_hour": "only timed stages (--hold) report tasks_per_hour",
        "shed_rate": "only open-loop runs (--rate) shed arrivals",
        "error_rate": "no model call reached the proxy",
        "p50_run_s": "no execution finished",
        "p95_run_s": "no execution finished",
    }
    if summary.executions == 0:
        reasons["cost_per_pass"] = "no execution finished"
    return {
        name: reasons.get(name, f"no execution reported {name}")
        for name, estimate in metrics.items()
        if estimate.value is None
    }


def analyze(run_dir: Path, options: RunOptions, *, baseline: Baseline | None = None) -> Analysis:
    """Everything the reports and gates need. Works on a run that is still being written."""
    data = RunData.load(run_dir)
    config = data.run.config if data.run else {}
    primary = str(config.get("primary", "passed"))
    seed = f"{options.seed}:analysis"
    steady = [e for e in data.finished() if not e.ramping]
    first = [e for e in steady if e.repeat == 0]
    repeats = [e for e in steady if e.repeat > 0]
    summary = group(data, steady, primary, seed=f"{seed}:all")
    stages = stage_stats(data, primary, seed=seed)
    metrics = _gate_metrics(data, steady, summary, stages, seed)
    percent = frozenset(n for n in metrics if is_bool_metric(data, n) or n.endswith("_rate"))
    rows = _task_rows(data, data.finished(), primary)
    versions = config.get("versions")
    models = {c.model for c in data.calls if c.model and c.role == "agent"}
    analysis = Analysis(
        run_id=data.run.run_id if data.run else run_dir.name,
        project=data.run.project if data.run else run_dir.parent.name,
        primary=primary,
        summary=summary,
        first_attempts=group(data, first, primary, seed=f"{seed}:first"),
        repeats=group(data, repeats, primary, seed=f"{seed}:repeats") if repeats else None,
        stages=stages,
        breaking_point=breaking_point(stages, options.breaking_p95),
        series=time_series(data, primary, options.interval_s),
        tasks=rows,
        metrics=metrics,
        percent_metrics=percent,
        started_at=baselines.iso(data.run.started_ns) if data.run else "",
        seed=int(str(config.get("seed", options.seed))),
        models=sorted(models),
        evaluation_version=versions.get("evaluation") if isinstance(versions, dict) else None,
        could_not_run=could_not_run(data),
        cost_note=data.cost_note(),
        cost_status=str(data.cost_status()),
        unpriced_models=data.unpriced_models(),
        usage_unknown=data.usage_unknown(),
        token_cap=_token_cap(config),
        notes=_notes(data, summary, metrics),
    )
    gate_metrics, against = metrics, baseline.summary if baseline else None
    comparison = baselines.compare(analysis, baseline) if baseline else None
    if baseline and comparison and (comparison.only_now or comparison.only_baseline):
        matched = set(comparison.matched)
        shared = [e for e in steady if e.task_id in matched]
        gate_metrics = {**metrics, primary: primary_estimate(data, shared, primary, seed=seed)}
        against = {**baseline.summary, primary: baseline.primary_over(comparison.matched)}
    try:
        gates = evaluate(
            parse_all(options.fail_if),
            gate_metrics,
            baseline=against,
            percent_metrics=percent,
            notes=analysis.notes,
        )
        gate_error = None
    except ConfigError as exc:
        gates, gate_error = [], str(exc)
    if comparison and (comparison.only_now or comparison.only_baseline):
        unmatched = len(comparison.only_now) + len(comparison.only_baseline)
        for gate in parse_all(options.fail_if):
            if gate.uses_baseline and gate.metric != primary:
                comparison.warnings.append(
                    f"gate {gate.text!r} compares all tasks, {unmatched} of them unmatched"
                )
    ab = config.get("ab")
    arms = None
    if isinstance(ab, dict):
        declared = config.get("metrics", {})
        spec = declared.get(primary, {}) if isinstance(declared, dict) else {}
        arms = compare_arms(
            data,
            variable=ab["var"],
            a=ab["a"],
            b=ab["b"],
            spec=MetricSpec(
                name=primary,
                direction=Direction(spec.get("direction", "higher")),
                worst=spec.get("worst"),
            ),
            seed=seed,
        )
    return copy.replace(
        analysis, gates=gates, comparison=comparison, gate_error=gate_error, ab=arms
    )


__all__ = ["Analysis", "TaskRow", "analyze"]
