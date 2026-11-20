"""Agentfile users for telemetry tests: one emits OpenTelemetry spans, one emits none."""

from __future__ import annotations

import os

from opentelemetry import trace

from instar import AgentUser, Context, Task
from tests.fixtures.agents import answer, traced_answer

TASKS = (
    Task(id="calm", data={"tools": ["lookup"]}),
    Task(id="loop", data={"tools": ["lookup"] * 5}),
    Task(id="forbidden", data={"tools": ["delete_file"]}),
    Task(id="long", data={"tools": [f"tool{i}" for i in range(6)]}),
)


class TracedBot(AgentUser):
    """Its model spans report TRACE_COST each; its judge emits a span that must not count."""

    tasks = TASKS

    async def run(self, task: Task, ctx: Context) -> str:
        cost = float(os.environ.get("TRACE_COST", "5e-6"))
        tools = [str(tool) for tool in task.data["tools"]]  # ty: ignore[not-iterable]  task data is JSON
        return await traced_answer(task.id, tools=tools, cost_usd=cost)

    async def judge(self, task: Task, output: str) -> bool:
        tracer = trace.get_tracer(__name__)
        attributes = {"gen_ai.operation.name": "chat", "test.cost_usd": 1.0}
        with tracer.start_as_current_span("chat judge", attributes=attributes):
            return output == f"echo: {task.id}"


class QuietBot(AgentUser):
    tasks = TASKS

    def run(self, task: Task, ctx: Context) -> str:
        return answer(task.id)

    def judge(self, task: Task, output: str) -> bool:
        return output == f"echo: {task.id}"
