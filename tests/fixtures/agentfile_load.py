"""A fast async agent for load tests: one model call per task through ctx.model_url."""

from __future__ import annotations

from instar import AgentUser, Context, Task
from tests.fixtures.agents import answer_async


class LoadBot(AgentUser):
    tasks = tuple(Task(id=f"t{i:02d}", data={"prompt": f"load {i}"}) for i in range(16))

    async def run(self, task: Task, ctx: Context) -> str:
        return await answer_async(str(task.prompt), base_url=ctx.model_url("openai"))

    def judge(self, task: Task, output: str) -> bool:
        return output == f"echo: {task.prompt}"
