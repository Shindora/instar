from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from instar.config import ConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from instar.analyze.stats import Estimate

EXPRESSION = re.compile(
    r"^\s*(?P<metric>[A-Za-z_][\w.]*)\s*(?P<op><=|>=|==|!=|<|>)\s*"
    r"(?:(?P<number>[-+]?\d+(?:\.\d+)?)"
    r"|baseline(?:\s*(?P<adjust_op>[-+*])\s*(?P<adjust>\d+(?:\.\d+)?))?)\s*$"
)
COMPARE: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
ADJUST: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}
WORDS = {
    "<": "Fails below",
    "<=": "Fails at or below",
    ">": "Fails above",
    ">=": "Fails at or above",
    "==": "Fails at",
    "!=": "Fails away from",
}


class GateState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True, kw_only=True)
class Gate:
    """One `--fail-if` comparison: `<metric> <op> <number>` or `<metric> <op> baseline [+-* n]`."""

    text: str
    metric: str
    op: str
    number: float | None
    adjust_op: str | None
    adjust: float

    @classmethod
    def parse(cls, text: str) -> Gate:
        match = EXPRESSION.match(text)
        if not match:
            raise ConfigError(
                f"cannot read --fail-if {text!r}: use '<metric> <op> <number>'"
                " or '<metric> <op> baseline [+|-|* <number>]'"
            )
        return cls(
            text=text.strip(),
            metric=match["metric"],
            op=match["op"],
            number=float(match["number"]) if match["number"] else None,
            adjust_op=match["adjust_op"],
            adjust=float(match["adjust"] or 0),
        )

    @property
    def uses_baseline(self) -> bool:
        return self.number is None

    def threshold(self, baseline: Mapping[str, float | None] | None) -> float | None:
        """The number to compare with; None when the baseline has no value for the metric."""
        if self.number is not None:
            return self.number
        if baseline is None:
            raise ConfigError(f"gate {self.text!r} compares with a baseline, but none was given")
        before = baseline.get(self.metric)
        if before is None:
            return None
        return ADJUST[self.adjust_op](before, self.adjust) if self.adjust_op else before


@dataclass(frozen=True, slots=True, kw_only=True)
class GateResult:
    text: str
    metric: str
    op: str
    threshold: float | None
    value: float | None
    low: float | None
    high: float | None
    state: GateState
    percent: bool
    reason: str | None = None
    lower_bound: bool = False

    @property
    def sentence(self) -> str:
        """For example: "Fails below 83%. Now 86%, likely between 79% and 91%."."""
        show = _percent if self.percent else _number
        rule = "Fails" if self.threshold is None else f"{WORDS[self.op]} {show(self.threshold)}."
        if self.value is None or self.threshold is None:
            return f"{rule} Not evaluated: {self.reason}."
        now = f"Now {'at least ' if self.lower_bound else ''}{show(self.value)}"
        if self.low is not None and self.high is not None:
            now += f", likely between {show(self.low)} and {show(self.high)}"
        verdict = {
            GateState.PASSED: "",
            GateState.FAILED: " Failed.",
            GateState.INCONCLUSIVE: " Inconclusive: the threshold lies inside the interval.",
            GateState.NOT_EVALUATED: f" Not evaluated: {self.reason}.",
        }[self.state]
        return f"{rule} {now}.{verdict}"


def _percent(value: float) -> str:
    return f"{value:.0%}"


def _number(value: float) -> str:
    return f"{value:.3g}"


def parse_all(texts: Sequence[str]) -> list[Gate]:
    return [Gate.parse(text) for text in texts]


def evaluate(
    gates: Sequence[Gate],
    metrics: Mapping[str, Estimate],
    *,
    baseline: Mapping[str, float | None] | None,
    percent_metrics: frozenset[str],
    notes: Mapping[str, str] | None = None,
) -> list[GateResult]:
    """A metric with no value, or no baseline value, makes its gate "not evaluated": it never
    passes silently. A metric name the run does not know is a configuration error."""
    results = []
    for gate in gates:
        if gate.metric not in metrics:
            known = ", ".join(sorted(metrics))
            raise ConfigError(f"gate {gate.text!r}: no metric {gate.metric} (have: {known})")
        estimate = metrics[gate.metric]
        threshold = gate.threshold(baseline)
        reason = None
        if estimate.value is None:
            reason = (notes or {}).get(gate.metric, f"{gate.metric} has no value in this run")
        elif threshold is None:
            reason = f"the baseline has no value for {gate.metric}"
        if reason is not None or estimate.value is None or threshold is None:
            state = GateState.NOT_EVALUATED
        elif estimate.lower_bound:
            state, reason = _from_lower_bound(gate.op, estimate.value, threshold, gate.metric)
        elif gate.op in {"<", "<=", ">", ">="} and estimate.contains(threshold):
            state = GateState.INCONCLUSIVE
        elif COMPARE[gate.op](estimate.value, threshold):
            state = GateState.FAILED
        else:
            state = GateState.PASSED
        results.append(
            GateResult(
                text=gate.text,
                metric=gate.metric,
                op=gate.op,
                threshold=threshold,
                value=estimate.value,
                low=estimate.low,
                high=estimate.high,
                state=state,
                percent=gate.metric in percent_metrics,
                reason=reason,
                lower_bound=estimate.lower_bound,
            )
        )
    return results


def _from_lower_bound(
    op: str, value: float, threshold: float, metric: str
) -> tuple[GateState, str | None]:
    """The true value is at least `value`: decide only what that bound alone proves."""
    if op in {">", ">="} and COMPARE[op](value, threshold):
        return GateState.FAILED, None
    if op in {"<", "<="} and not COMPARE[op](value, threshold):
        return GateState.PASSED, None
    return GateState.NOT_EVALUATED, f"{metric} is only a lower bound"


def gate_exit_code(results: Sequence[GateResult], *, strict: bool) -> int:
    """1 when a gate failed; under --strict also when one was inconclusive or not evaluated."""
    failing = {GateState.FAILED}
    if strict:
        failing |= {GateState.INCONCLUSIVE, GateState.NOT_EVALUATED}
    return 1 if any(r.state in failing for r in results) else 0
