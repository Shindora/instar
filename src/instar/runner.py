from __future__ import annotations

import asyncio
import copy
import json
import os
import platform
import secrets
import tempfile
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from instar.dataset import limit
from instar.events import Execution, ExecutionStatus, Metric, Run
from instar.executors import ExecutionSpec, Outcome, executors
from instar.proxy.app import Proxy
from instar.proxy.upstreams import DEFAULT_MODEL_ENV, model_url, resolve_upstreams
from instar.server import serve_in_thread
from instar.store import Store
from instar.user import load_agentfile

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from instar.dataset import Task
    from instar.executors.base import ExecutorFn
    from instar.user import AgentUser

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_USERS = 4
MAX_PROCESS_USERS = 64
DEFAULT_ARM = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOptions:
    agentfile: Path
    project: str
    users: int = DEFAULT_USERS
    repeat: int = 1
    limit: int | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    executor: str = "process"
    user: str | None = None
    seed: int = 0
    faults: dict[str, float] = field(default_factory=dict)
    latency_s: float = 5.0
    spend_cap_usd: float | None = None
    keep_outputs: bool = False
    runs_dir: Path = Path("runs")
    upstreams: dict[str, str] = field(default_factory=dict)
    model_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult:
    run_id: str
    run_dir: Path
    statuses: dict[ExecutionStatus, int]
    primary: str
    cancelled: bool


def new_session() -> str:
    """A W3C trace id and span id, so spans can be joined to the execution."""
    return f"{secrets.token_hex(16)}-{secrets.token_hex(8)}"


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def _resource_attributes(session: str) -> str:
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
    attribute = f"instar.session={session}"
    return f"{existing},{attribute}" if existing else attribute


@contextmanager
def _sessionless_env(model_env: Mapping[str, str], proxy_url: str) -> Generator[None]:
    """Point model variables at the proxy for the whole run, without a session.

    Only for the inline executor: an agent that reads them still reaches the proxy, and
    `doctor` can say its calls arrived without a session. This is run-wide, never per execution.
    """
    saved = {var: os.environ.get(var) for var in model_env}
    os.environ.update({var: f"{proxy_url}/{name}" for var, name in model_env.items()})
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


class Runner:
    """Closed loop: `users` concurrent users each take the next (task, repeat) until none is left.

    One run is one proxy and one store. `stop()` starts no new execution, cancels those in flight
    (recorded as `cancelled`) and still closes the run cleanly.
    """

    def __init__(self, options: RunOptions) -> None:
        self.options = options
        self.warnings: list[str] = []
        self.users = options.users
        if options.executor == "process" and options.users > MAX_PROCESS_USERS:
            self.users = MAX_PROCESS_USERS
            self.warnings.append(
                f"users capped at {MAX_PROCESS_USERS}: each process user is a Python process"
                " holding the agent's imports"
            )
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        self.run_dir = options.runs_dir / options.project / self.run_id
        self.statuses: Counter[ExecutionStatus] = Counter()
        self.primary = "passed"
        self._stopping = False
        self._user_tasks: list[asyncio.Task[None]] = []

    def stop(self) -> None:
        self._stopping = True
        for task in self._user_tasks:
            task.cancel()

    def _config(self) -> dict[str, object]:
        return {
            **asdict(self.options),
            "users": self.users,
            "run_id": self.run_id,
            "versions": {
                "instar": metadata.version("instar"),
                "python": platform.python_version(),
            },
        }

    async def run(self) -> RunResult:
        options = self.options
        executor = executors[options.executor]
        store = Store(self.run_dir)
        config = self._config()
        _write_json(self.run_dir / "config.json", config)
        run = Run(run_id=self.run_id, project=options.project, started_ns=time.time_ns())
        store.emit(copy.replace(run, config=config))
        upstreams = resolve_upstreams(options.upstreams, options.model_env)
        model_env = {
            var: name
            for var, name in {**DEFAULT_MODEL_ENV, **options.model_env}.items()
            if name in upstreams
        }
        proxy = Proxy(
            upstreams=upstreams,
            sink=store,
            seed=options.seed,
            fault_rates=options.faults,
            latency_s=options.latency_s,
            spend_cap_usd=options.spend_cap_usd,
        )
        try:
            with serve_in_thread(proxy.build_app()) as proxy_url:
                inline = options.executor == "inline"
                with _sessionless_env(model_env if inline else {}, proxy_url):
                    user = load_agentfile(options.agentfile, options.user)
                    self.primary = user.primary
                    tasks = limit(user.tasks, options.limit)
                    work = deque((t, r) for r in range(options.repeat) for t in tasks)
                    loop = _Loop(self, user, executor, proxy, proxy_url, model_env, store)
                    async with asyncio.TaskGroup() as group:
                        self._user_tasks = [
                            group.create_task(loop.user(work))
                            for _ in range(min(self.users, len(work)))
                        ]
                        if self._stopping:
                            self.stop()
        finally:
            store.emit(copy.replace(run, config=config, ended_ns=time.time_ns()))
            store.close()
        return RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            statuses=dict(self.statuses),
            primary=self.primary,
            cancelled=self._stopping,
        )


@dataclass(slots=True)
class _Loop:
    runner: Runner
    user_class: type[AgentUser]
    executor: ExecutorFn
    proxy: Proxy
    proxy_url: str
    model_env: dict[str, str]
    store: Store

    async def user(self, work: deque[tuple[Task, int]]) -> None:
        while work and not self.runner._stopping:
            task, repeat = work.popleft()
            await self.execute(task, repeat)

    async def execute(self, task: Task, repeat: int) -> None:
        options = self.runner.options
        session = new_session()
        self.proxy.fault_keys[session] = f"{task.id}:{repeat}"
        env = {var: model_url(self.proxy_url, session, up) for var, up in self.model_env.items()}
        env["OTEL_RESOURCE_ATTRIBUTES"] = _resource_attributes(session)
        started_ns = time.time_ns()
        clock = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="instar-work-") as workdir:
            spec = ExecutionSpec(
                agentfile=options.agentfile,
                task=task,
                session=session,
                arm=DEFAULT_ARM,
                proxy_url=self.proxy_url,
                env=env,
                workdir=Path(workdir),
                log_path=self.runner.run_dir / "logs" / f"{session}.log",
                timeout_s=options.timeout_s,
                keep_output=options.keep_outputs,
            )
            try:
                outcome = await self.executor(spec, self.user_class)
            except asyncio.CancelledError:
                elapsed_ns = time.perf_counter_ns() - clock
                cancelled = Outcome(status=ExecutionStatus.CANCELLED, run_ns=elapsed_ns)
                self.record(session, task, repeat, started_ns, cancelled)
                raise
        self.record(session, task, repeat, started_ns, outcome)

    def record(
        self, session: str, task: Task, repeat: int, started_ns: int, outcome: Outcome
    ) -> None:
        self.runner.statuses[outcome.status] += 1
        self.store.emit(
            Execution(
                session=session,
                task_id=task.id,
                arm=DEFAULT_ARM,
                repeat=repeat,
                stage=1,
                status=outcome.status,
                started_ns=started_ns,
                run_ns=outcome.run_ns,
                judge_ns=outcome.judge_ns,
                error=outcome.error,
                output=outcome.output,
            )
        )
        for name, value in outcome.metrics.items():
            self.store.emit(Metric(session=session, name=name, value=value))
