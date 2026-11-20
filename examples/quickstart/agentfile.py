from pathlib import Path

from agent import answer

from instar import AgentUser, Context, Task, dataset


class Quickstart(AgentUser):
    tasks = dataset.from_jsonl(Path(__file__).with_name("testset.jsonl"))

    def run(self, task: Task, ctx: Context) -> str:
        return answer(str(task.prompt))

    def judge(self, task: Task, output: str) -> bool:
        return str(task.expected).lower() in output.lower()
