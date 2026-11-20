"""Child process for one execution: `python -m instar.executors.worker OUTCOME_PATH < spec.json`."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from instar.dataset import Task
from instar.events import ExecutionStatus
from instar.executors.base import Outcome, execute
from instar.user import AgentfileError, Context, load_agentfile


def _write_atomically(path: Path, text: str) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text(text)
    partial.replace(path)


def main(argv: list[str]) -> int:
    outcome_path = Path(argv[0])
    spec = json.loads(sys.stdin.read())
    try:
        user = load_agentfile(Path(spec["agentfile"]), spec["user"])
    except AgentfileError as exc:
        outcome = Outcome(status=ExecutionStatus.ERROR, error=str(exc))
    else:
        task = Task(id=spec["task"]["id"], data=spec["task"]["data"])
        ctx = Context(
            session=spec["session"],
            arm=spec["arm"],
            workdir=Path(spec["workdir"]),
            proxy_url=spec["proxy_url"],
        )
        outcome = asyncio.run(execute(user, task, ctx, keep_output=spec["keep_output"]))
    _write_atomically(outcome_path, outcome.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
