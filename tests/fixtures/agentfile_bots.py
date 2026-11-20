"""Well-behaved agentfile users: sync and async, and one that reads its base URL from the env."""

from __future__ import annotations

import asyncio

from instar import AgentUser, Context, Task
from tests.fixtures.agents import answer, answer_async

TASKS = tuple(Task(id=f"t{i}", data={"prompt": f"hello {i}"}) for i in range(3))


class SyncBot(AgentUser):
    tasks = TASKS

    def run(self, task: Task, ctx: Context) -> str:
        return answer(str(task.prompt), base_url=ctx.model_url("openai"))

    def judge(self, task: Task, output: str) -> bool:
        return output == f"echo: {task.prompt}"


class AsyncBot(AgentUser):
    tasks = TASKS

    async def run(self, task: Task, ctx: Context) -> str:
        base_url = ctx.model_url("anthropic")
        return await answer_async(str(task.prompt), provider="anthropic", base_url=base_url)

    async def judge(self, task: Task, output: str) -> dict[str, object]:
        await asyncio.sleep(0)
        return {"passed": output == f"echo: {task.prompt}", "length": len(output)}


class EnvBot(AgentUser):
    """Reads OPENAI_BASE_URL like most agents, so in inline mode its calls carry no session."""

    tasks = TASKS

    def run(self, task: Task, ctx: Context) -> str:
        return answer(str(task.prompt))
