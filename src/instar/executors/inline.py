from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from instar.events import ExecutionStatus
from instar.executors.base import Outcome, execute, executors
from instar.telemetry.root import flush

if TYPE_CHECKING:
    from instar.executors.base import ExecutionSpec
    from instar.user import AgentUser


@executors("inline")
async def run_inline(spec: ExecutionSpec, user: type[AgentUser]) -> Outcome:
    """Run in the runner's own process. Settings arrive only through `ctx`, never the env.

    ponytail: a hung sync `run` keeps its thread after the timeout; Python cannot kill a
    thread. Use the process executor for agents that may hang.
    """
    clock = time.perf_counter_ns()
    try:
        async with asyncio.timeout(spec.timeout_s):
            try:
                return await execute(user, spec.task, spec.context(), keep_output=spec.keep_output)
            finally:
                await asyncio.to_thread(flush)
    except TimeoutError:
        return Outcome(status=ExecutionStatus.TIMEOUT, run_ns=time.perf_counter_ns() - clock)
