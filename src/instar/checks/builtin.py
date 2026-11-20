from __future__ import annotations

import itertools

from instar.checks import CALLS, SPANS, Evidence, Verdict, check, failed, not_evaluated, passed
from instar.events import ExecutionStatus, SpanKind

STOPPED_BY_INSTAR = frozenset({ExecutionStatus.TIMEOUT, ExecutionStatus.CRASHED})


@check("terminated", needs=())
def terminated(evidence: Evidence, setting: object) -> Verdict:
    """The agent ended on its own: no timeout, no crash."""
    status = evidence.execution.status
    if status is ExecutionStatus.CANCELLED:
        return not_evaluated("the run was cancelled")
    if status in STOPPED_BY_INSTAR:
        return failed(f"the execution ended as {status}")
    return passed(f"ended on its own ({status})")


@check("max_llm_calls", needs=(CALLS,), default=50)
def max_llm_calls(evidence: Evidence, limit: int) -> Verdict:
    count = len(evidence.calls)
    if count > limit:
        return failed(f"{count} model calls, more than {limit}")
    return passed(f"{count} model calls")


@check("cost_cap", needs=(CALLS,), default=None)
def cost_cap(evidence: Evidence, limit_usd: float) -> Verdict:
    if any(call.cost_usd is None for call in evidence.calls):
        return not_evaluated("cost unknown: a call has no price or no usage")
    spent = sum(call.cost_usd or 0.0 for call in evidence.calls)
    if spent > limit_usd:
        return failed(f"${spent:.6f}, more than ${limit_usd:g}")
    return passed(f"${spent:.6f}")


@check("backoff_after_429", needs=(CALLS,), default=1.0)
def backoff_after_429(evidence: Evidence, min_wait_s: float) -> Verdict:
    """After a 429, the next model call waits at least `min_wait_s`."""
    calls = sorted(evidence.calls, key=lambda c: c.index)
    gaps = [
        (later.started_ns - earlier.ended_ns) / 1e9
        for earlier, later in itertools.pairwise(calls)
        if earlier.status == 429
    ]
    if not gaps:
        return not_evaluated("no 429 was followed by another call")
    if min(gaps) < min_wait_s:
        return failed(f"retried {min(gaps):.2f} s after a 429, sooner than {min_wait_s:g} s")
    return passed(f"waited at least {min(gaps):.2f} s after each 429")


@check("max_steps", needs=(SPANS,), default=50)
def max_steps(evidence: Evidence, limit: int) -> Verdict:
    """Steps are model and tool spans."""
    steps = sum(1 for span in evidence.spans if span.kind in {SpanKind.LLM, SpanKind.TOOL})
    if steps > limit:
        return failed(f"{steps} steps, more than {limit}")
    return passed(f"{steps} steps")


@check("no_tool_loop", needs=(SPANS,), default=4)
def no_tool_loop(evidence: Evidence, repeats: int) -> Verdict:
    """The same tool with the same arguments, `repeats` times in a row, is a loop."""
    run, longest, previous = 0, (0, ""), None
    for tool in evidence.tools():
        run = run + 1 if tool == previous else 1
        previous = tool
        longest = max(longest, (run, tool[0]))
    if longest[0] >= repeats:
        return failed(f"{longest[1]} called {longest[0]} times in a row with the same arguments")
    return passed(f"longest repeat: {longest[0]}")


@check("forbidden_tools", needs=(SPANS,), default=None)
def forbidden_tools(evidence: Evidence, names: list[str]) -> Verdict:
    used = sorted({name for name, _ in evidence.tools()} & set(names))
    if used:
        return failed(f"called {', '.join(used)}")
    return passed("no forbidden tool called")
