"""Agentfile users whose behaviour depends on AGENT_VERSION, for A/B tests."""

from __future__ import annotations

import os

import httpx

from instar import AgentUser, Context, Task
from tests.fixtures.agents import answer


class VersionBot(AgentUser):
    """Under any AGENT_VERSION starting with v1 it fails the even tasks; otherwise it passes all."""

    tasks = tuple(
        Task(id=f"t{i:02d}", data={"prompt": f"ab {i}", "hard": i % 2 == 0}) for i in range(20)
    )

    def run(self, task: Task, ctx: Context) -> str:
        reply = answer(str(task.prompt))
        if task.hard and os.environ.get("AGENT_VERSION", "").startswith("v1"):
            return "wrong"
        return reply

    def judge(self, task: Task, output: str) -> bool:
        return output == f"echo: {task.prompt}"


class StubbornBot(AgentUser):
    """Makes three calls and carries on when one fails, so every call gets a fault decision."""

    tasks = tuple(Task(id=f"t{i:02d}", data={}) for i in range(10))

    def run(self, task: Task, ctx: Context) -> str:
        replies = []
        for _ in range(3):
            try:
                replies.append(answer(task.id))
            except httpx.HTTPStatusError as exc:
                replies.append(str(exc.response.status_code))
        return ",".join(replies)
