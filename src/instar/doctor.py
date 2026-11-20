from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from instar.checks import SPANS, checks
from instar.events import CheckResult, ExecutionStatus, LLMCall, Metric, Span
from instar.events import Execution as ExecutionEvent
from instar.store import StoreReader
from instar.telemetry.receiver import available
from instar.telemetry.root import split_phases

if TYPE_CHECKING:
    from instar.runner import RunOptions, RunResult

TIMEOUT_HEADROOM = 3


@dataclass(slots=True)
class Diagnosis:
    lines: list[str] = field(default_factory=list)
    problems: int = 0

    def note(self, line: str) -> None:
        self.lines.append(line)

    def problem(self, line: str) -> None:
        self.problems += 1
        self.lines.append(f"PROBLEM: {line}")

    @property
    def exit_code(self) -> int:
        return 3 if self.problems else 0


def _telemetry(
    found: Diagnosis, session: str, spans: list[Span], verdicts: list[CheckResult]
) -> None:
    """Which spans arrived, which checks could therefore run, and what they found."""
    span_checks = sorted(name for name in checks.names() if SPANS in checks[name].needs)
    agent, judge = split_phases([s for s in spans if s.session == session], session)
    if not available():
        found.note("telemetry: spans cannot be received without the otel extra (instar[otel])")
    elif agent:
        kinds = ", ".join(
            f"{k} {n}" for k, n in sorted(Counter(str(s.kind) for s in agent).items())
        )
        found.note(
            f"spans for this execution: {kinds}; judge spans kept out of checks: {len(judge)}"
        )
    else:
        found.note(
            "no spans arrived for this execution: span checks"
            f" ({', '.join(span_checks)}) report not evaluated"
        )
    unjoined = sum(1 for s in spans if not s.session)
    found.note(f"spans not joined to any execution: {unjoined}")
    for verdict in sorted(verdicts, key=lambda v: v.check):
        found.note(f"check {verdict.check}: {verdict.status} ({verdict.detail})")


def _no_calls_hint(executor: str) -> str:
    if executor == "inline":
        return "In inline mode, pass ctx.model_url(name) to the agent's model client."
    return (
        "The agent may set its base URL in code (for example http_options.base_url), which"
        " overrides the environment, or call a provider instar has no upstream for."
    )


def diagnose(options: RunOptions, result: RunResult) -> Diagnosis:
    """Explain what one preflight execution did and what arrived from it."""
    found = Diagnosis()
    reader = StoreReader(result.run_dir)
    try:
        [execution] = reader.select(ExecutionEvent)
        calls = reader.select(LLMCall)
        metrics = reader.select(Metric, session=execution.session)
        spans = reader.select(Span)
        verdicts = reader.select(CheckResult, session=execution.session)
    finally:
        reader.close()
    run_s = execution.run_ns / 1e9
    found.note(f"run {result.run_id} with the {options.executor} executor: {result.run_dir}")
    found.note(f"task {execution.task_id}: {execution.status} after {run_s:.2f} s")
    if execution.status is not ExecutionStatus.OK:
        last_line = (execution.error or "").strip().splitlines()[-1:] or ["no detail"]
        found.problem(f"the task ended as {execution.status}: {last_line[0]}")

    own = [c for c in calls if c.session == execution.session]
    sessionless = [c for c in calls if c.session == ""]
    if own:
        upstreams = ", ".join(
            f"{u} x{n}" for u, n in sorted(Counter(c.upstream for c in own).items())
        )
        tokens_in = sum(c.input_tokens for c in own)
        tokens_out = sum(c.output_tokens for c in own)
        found.note(
            f"model calls attributed to this session: {len(own)} ({upstreams}),"
            f" {tokens_in} tokens in, {tokens_out} out"
        )
        failed = Counter(c.status for c in own if c.status >= 400 and c.fault is None)
        if failed:
            codes = ", ".join(f"HTTP {code} x{n}" for code, n in sorted(failed.items()))
            found.problem(f"model calls failed upstream: {codes}; check credentials and quotas")
    else:
        found.problem(f"no calls arrived for this session. {_no_calls_hint(options.executor)}")
    if sessionless:
        found.problem(
            f"{len(sessionless)} model calls arrived without a session. Pass"
            " ctx.model_url(name) to the agent in inline mode, or use the process executor."
        )

    if own:
        priced = [c.cost_usd for c in own]
        if None in priced:
            notes = sorted({c.cost_note or "cost unknown" for c in own if c.cost_usd is None})
            found.note(f"cost: unknown ({'; '.join(notes)}; add it under [tool.instar.prices])")
        else:
            found.note(f"cost: ${sum(c or 0.0 for c in priced):.6f} from the price table")
    if metrics:
        found.note("judge: " + ", ".join(f"{m.name}={m.value}" for m in metrics))
    elif execution.status is ExecutionStatus.OK:
        found.note("judge: none (the agentfile defines no judge)")

    _telemetry(found, execution.session, spans, verdicts)

    if execution.status is ExecutionStatus.TIMEOUT:
        found.problem(f"the task did not finish within timeout_s = {options.timeout_s:g} s")
    elif execution.status is ExecutionStatus.OK:
        needed_s = TIMEOUT_HEADROOM * run_s
        if options.timeout_s < needed_s:
            found.note(
                f"WARNING: timeout_s = {options.timeout_s:g} s is below {TIMEOUT_HEADROOM}x the"
                f" measured run time ({needed_s:.1f} s); raise it with --timeout-s"
            )
        else:
            found.note(
                f"timeout_s = {options.timeout_s:g} s leaves room"
                f" ({TIMEOUT_HEADROOM}x run time = {needed_s:.1f} s)"
            )
    return found
