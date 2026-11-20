"""Agentfile users that misbehave on purpose. They read their base URL from the environment."""

from __future__ import annotations

import time

from instar import AgentUser, Context, Task
from tests.fixtures.agents import answer, exiting, import_time_answer, raising, stalling


class FaultyBot(AgentUser):
    tasks = tuple(Task(id=kind, data={"kind": kind}) for kind in ("ok", "raise", "hang", "exit"))

    def run(self, task: Task, ctx: Context) -> str:
        match task.kind:
            case "raise":
                return raising("x")
            case "hang":
                return stalling("x")
            case "exit":
                return exiting("x")
            case _:
                return answer("x")


class ImportTimeBot(AgentUser):
    tasks = tuple(Task(id=f"t{i}", data={}) for i in range(6))

    def run(self, task: Task, ctx: Context) -> str:
        return import_time_answer(task.id)


class HangBot(AgentUser):
    tasks = tuple(Task(id=f"t{i}", data={}) for i in range(3))

    def run(self, task: Task, ctx: Context) -> str:
        return stalling(task.id)


class SlowBot(AgentUser):
    tasks = (Task(id="slow", data={}),)

    def run(self, task: Task, ctx: Context) -> str:
        time.sleep(0.6)
        return answer("x")
