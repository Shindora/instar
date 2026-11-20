from __future__ import annotations

import asyncio
import inspect
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from instar.events import ExecutionStatus
from instar.judge import as_metrics
from instar.registry import Registry
from instar.telemetry.root import phase
from instar.user import Context

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from instar.dataset import Task
    from instar.user import AgentUser

type MetricValue = bool | int | float | str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """What one execution produced. Process workers send it back as JSON through a file."""

    status: ExecutionStatus
    run_ns: int = 0
    judge_ns: int | None = None
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    output: object = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, text: str) -> Outcome:
        data = json.loads(text)
        return cls(
            status=ExecutionStatus(data["status"]),
            run_ns=data["run_ns"],
            judge_ns=data["judge_ns"],
            metrics=data["metrics"],
            output=data["output"],
            error=data["error"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionSpec:
    """Everything an executor needs to run one execution."""

    agentfile: Path
    task: Task
    session: str
    arm: str
    proxy_url: str
    env: dict[str, str]
    workdir: Path
    log_path: Path
    timeout_s: float
    keep_output: bool

    def context(self) -> Context:
        return Context(
            session=self.session, arm=self.arm, workdir=self.workdir, proxy_url=self.proxy_url
        )


type ExecutorFn = Callable[[ExecutionSpec, type[AgentUser]], Awaitable[Outcome]]

executors: Registry[ExecutorFn] = Registry("executor")


async def call_user(fn: Callable[..., object], *args: object) -> object:
    """Await an async function; run a sync one in a thread."""
    if inspect.iscoroutinefunction(fn):
        result = await fn(*args)
    else:
        result = await asyncio.to_thread(fn, *args)
    return await result if inspect.isawaitable(result) else result


def _metric_value(value: object) -> MetricValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


async def execute(user: type[AgentUser], task: Task, ctx: Context, *, keep_output: bool) -> Outcome:
    """Run `run`, then `judge` on its output, timing each. Agent exceptions become `error`."""
    agent = user()
    clock = time.perf_counter_ns()
    try:
        with phase(ctx.session, "run"):
            output = await call_user(agent.run, task, ctx)
    except Exception:
        return Outcome(
            status=ExecutionStatus.ERROR,
            run_ns=time.perf_counter_ns() - clock,
            error=traceback.format_exc(),
        )
    run_ns = time.perf_counter_ns() - clock
    kept = output if keep_output else None
    if not user.has_judge():
        return Outcome(status=ExecutionStatus.OK, run_ns=run_ns, output=kept)
    clock = time.perf_counter_ns()
    try:
        with phase(ctx.session, "judge"):
            metrics = as_metrics(await call_user(agent.judge, task, output))
    except Exception:
        return Outcome(
            status=ExecutionStatus.ERROR,
            run_ns=run_ns,
            judge_ns=time.perf_counter_ns() - clock,
            output=kept,
            error=f"judge failed:\n{traceback.format_exc()}",
        )
    return Outcome(
        status=ExecutionStatus.OK,
        run_ns=run_ns,
        judge_ns=time.perf_counter_ns() - clock,
        metrics={name: _metric_value(value) for name, value in metrics.items()},
        output=kept,
    )
