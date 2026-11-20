from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING

from instar.events import ExecutionStatus
from instar.executors.base import Outcome, executors

if TYPE_CHECKING:
    from instar.executors.base import ExecutionSpec
    from instar.user import AgentUser


def worker_input(spec: ExecutionSpec, user: type[AgentUser]) -> bytes:
    return json.dumps(
        {
            "agentfile": str(spec.agentfile),
            "user": user.__name__,
            "task": {"id": spec.task.id, "data": dict(spec.task.data)},
            "session": spec.session,
            "arm": spec.arm,
            "workdir": str(spec.workdir),
            "proxy_url": spec.proxy_url,
            "keep_output": spec.keep_output,
        }
    ).encode()


def _open_log(path: Path) -> IO[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")


def _read_outcome(path: Path) -> Outcome | None:
    return Outcome.from_json(path.read_text()) if path.exists() else None


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill the worker and anything the agent started in its process group."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


@executors("process")
async def run_in_process(spec: ExecutionSpec, user: type[AgentUser]) -> Outcome:
    """One fresh Python process per execution, with this execution's own environment.

    The worker reports through a file, so agent code may print freely; stdout and stderr go to
    the execution's log file.
    """
    env = {**os.environ, **spec.env, "PYTHONUNBUFFERED": "1"}
    with tempfile.TemporaryDirectory(prefix="instar-") as tmp, _open_log(spec.log_path) as log:
        outcome_path = Path(tmp) / "outcome.json"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "instar.executors.worker",
            str(outcome_path),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        clock = time.perf_counter_ns()
        try:
            async with asyncio.timeout(spec.timeout_s):
                await proc.communicate(worker_input(spec, user))
        except TimeoutError:
            await _kill(proc)
            return Outcome(status=ExecutionStatus.TIMEOUT, run_ns=time.perf_counter_ns() - clock)
        except asyncio.CancelledError:
            await _kill(proc)
            raise
        outcome = _read_outcome(outcome_path)
    if outcome is None:
        return Outcome(
            status=ExecutionStatus.CRASHED,
            run_ns=time.perf_counter_ns() - clock,
            error=f"worker exited with code {proc.returncode} before reporting an outcome",
        )
    return outcome
