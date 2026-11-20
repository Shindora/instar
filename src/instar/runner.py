from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import platform
import random
import secrets
import subprocess
import tempfile
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from instar.config import AB, Stage
from instar.dataset import limit
from instar.events import Execution, ExecutionStatus, Load, Metric, Run, Shed, StageWindow
from instar.executors import ExecutionSpec, Outcome, executors
from instar.proxy.app import Proxy
from instar.proxy.upstreams import DEFAULT_MODEL_ENV, model_url, resolve_upstreams
from instar.server import serve_in_thread
from instar.store import Store
from instar.user import AgentfileError, load_agentfile

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Generator, Mapping, Sequence

    from instar.dataset import Task
    from instar.executors.base import ExecutorFn
    from instar.proxy.pricing import Price
    from instar.user import AgentUser

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_USERS = 4
MAX_PROCESS_USERS = 64
DEFAULT_ARM = "default"


@dataclass(frozen=True, slots=True, kw_only=True)
class Arm:
    name: str
    env: dict[str, str]


type WorkItem = tuple[Task, int, Arm]
type ArmOrder = Callable[[Task, int], list[Arm]]


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOptions:
    agentfile: Path = Path("agentfile.py")
    project: str = "default"
    users: int = DEFAULT_USERS
    stages: tuple[Stage, ...] | None = None
    hold_s: float | None = None
    ramp_per_min: float | None = None
    think_time_s: float = 0.0
    rate_per_s: float | None = None
    max_users: int | None = None
    repeat: int = 1
    limit: int | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    executor: str = "process"
    user: str | None = None
    seed: int = 0
    faults: dict[str, float] = field(default_factory=dict)
    latency_s: float = 5.0
    spend_cap_usd: float | None = None
    token_cap: int | None = None
    prices: dict[str, Price] = field(default_factory=dict)
    keep_outputs: bool = False
    runs_dir: Path = Path("runs")
    upstreams: dict[str, str] = field(default_factory=dict)
    model_env: dict[str, str] = field(default_factory=dict)
    fail_if: tuple[str, ...] = ()
    baseline: Path | None = None
    strict: bool = False
    interval_s: float = 30.0
    breaking_p95: float = 2.0
    ab: AB | None = None
    kind: str = "run"

    def arms(self) -> tuple[Arm, ...]:
        if self.ab is None:
            return (Arm(name=DEFAULT_ARM, env={}),)
        ab = self.ab
        return (Arm(name=ab.a, env={ab.var: ab.a}), Arm(name=ab.b, env={ab.var: ab.b}))

    def shape(self) -> tuple[Stage, ...]:
        """The stages to run: the given list, or one stage from `users` and `hold_s`."""
        return self.stages or (Stage(users=self.users, hold_s=self.hold_s),)

    @property
    def timed(self) -> bool:
        return self.rate_per_s is not None or any(s.hold_s is not None for s in self.shape())


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult:
    run_id: str
    run_dir: Path
    statuses: dict[ExecutionStatus, int]
    primary: str
    cancelled: bool


def new_run_id() -> str:
    """Sorts by start time, to the millisecond, so the newest run is last by name."""
    now_ns = time.time_ns()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now_ns / 1e9))
    return f"{stamp}-{now_ns // 1_000_000 % 1000:03d}-{secrets.token_hex(3)}"


async def _either(*waits: Coroutine[object, object, object]) -> None:
    """Return when the first of these finishes; cancel the rest."""
    tasks = [asyncio.ensure_future(wait) for wait in waits]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()


def ordered_arms(arms: Sequence[Arm], *, seed: int, task_id: str, repeat: int) -> list[Arm]:
    """Both arms of one (task, repeat), next to each other; which goes first is seeded, so
    neither arm always runs second against a provider cache the other warmed."""
    order = list(arms)
    random.Random(f"{seed}:arms:{task_id}:{repeat}").shuffle(order)
    return order


def _check_arms(loop: _Loop, ab: AB | None) -> dict[str, dict[str, str]]:
    """Each arm's variables, with the session left as a placeholder; arms may differ in one."""
    envs = {arm.name: loop.overlay("<session>", arm) for arm in loop.runner.options.arms()}
    if ab is not None:
        first, second = envs.values()
        differ = {k for k in first.keys() | second.keys() if first.get(k) != second.get(k)}
        if differ != {ab.var}:
            raise RuntimeError(f"arms must differ only in {ab.var}, not in {sorted(differ)}")
    return envs


def _git(folder: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(folder), *args], capture_output=True, text=True, timeout=10, check=True
    )
    return done.stdout.strip()


def evaluation_version(agentfile: Path) -> str:
    """The git commit holding the agentfile (`-dirty` with local changes), else its hash."""
    folder = agentfile.resolve().parent
    try:
        commit = _git(folder, "rev-parse", "--short=12", "HEAD")
        dirty = _git(folder, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        content = agentfile.read_bytes() if agentfile.is_file() else b""
        return f"sha256:{hashlib.sha256(content).hexdigest()[:12]}"
    return f"{commit}-dirty" if dirty else commit


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


class Source(Protocol):
    def take(self) -> WorkItem | None: ...

    def finished(self) -> None: ...


class Drain:
    """Every (task, repeat) once; `drained` is set when all have been taken and finished."""

    def __init__(self, items: Sequence[WorkItem]) -> None:
        self.items = deque(items)
        self.in_flight = 0
        self.drained = asyncio.Event()
        if not self.items:
            self.drained.set()

    def take(self) -> WorkItem | None:
        if not self.items:
            return None
        self.in_flight += 1
        return self.items.popleft()

    def finished(self) -> None:
        self.in_flight -= 1
        if not self.items and not self.in_flight:
            self.drained.set()


class Rounds:
    """Endless rounds over a seeded shuffle of the task set; the round number is the repeat."""

    def __init__(self, tasks: Sequence[Task], seed: str, arms: ArmOrder) -> None:
        self.tasks = list(tasks)
        self.rng = random.Random(seed)
        self.arms = arms
        self.queue: deque[WorkItem] = deque()
        self.round = -1

    def take(self) -> WorkItem:
        if not self.queue:
            self.round += 1
            order = self.tasks[:]
            self.rng.shuffle(order)
            self.queue.extend(
                (task, self.round, arm) for task in order for arm in self.arms(task, self.round)
            )
        return self.queue.popleft()

    def finished(self) -> None:
        pass


class Runner:
    """Runs an agentfile through a load shape. One run is one proxy and one store.

    `stop()` starts no new execution, cancels those in flight (recorded as `cancelled`) and
    still closes the run cleanly.
    """

    def __init__(self, options: RunOptions) -> None:
        self.warnings: list[str] = []
        if options.executor == "process":
            options = self._cap_process_users(options)
        if options.spend_cap_usd is not None and not options.prices:
            self.warnings.append(
                "--spend-cap counts priced calls only and no prices are configured, so it cannot"
                " stop this run; set [tool.instar.prices] or use --token-cap"
            )
        self.options = options
        self.run_id = new_run_id()
        self.run_dir = options.runs_dir / options.project / self.run_id
        self.statuses: Counter[ExecutionStatus] = Counter()
        self.primary = "passed"
        self.stopping = False
        self._shape: asyncio.Task[None] | None = None

    def _cap_process_users(self, options: RunOptions) -> RunOptions:
        shape = options.shape()
        if (
            max(s.users for s in shape) <= MAX_PROCESS_USERS
            and (options.max_users or 0) <= MAX_PROCESS_USERS
        ):
            return options
        self.warnings.append(
            f"users capped at {MAX_PROCESS_USERS}: each process user is a Python process"
            " holding the agent's imports"
        )
        capped = tuple(copy.replace(s, users=min(s.users, MAX_PROCESS_USERS)) for s in shape)
        max_users = None if options.max_users is None else min(options.max_users, MAX_PROCESS_USERS)
        return copy.replace(options, stages=capped, max_users=max_users)

    @property
    def users(self) -> int:
        return max(s.users for s in self.options.shape())

    def stop(self) -> None:
        self.stopping = True
        if self._shape is not None:
            self._shape.cancel()

    def _config(self) -> dict[str, object]:
        return {
            **asdict(self.options),
            "run_id": self.run_id,
            "versions": {
                "instar": metadata.version("instar"),
                "python": platform.python_version(),
                "evaluation": evaluation_version(self.options.agentfile),
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
            token_cap=options.token_cap,
            prices=options.prices,
        )
        try:
            with serve_in_thread(proxy.build_app()) as proxy_url:
                inline = options.executor == "inline"
                with _sessionless_env(model_env if inline else {}, proxy_url):
                    user = load_agentfile(options.agentfile, options.user)
                    self.primary = user.primary
                    specs = user.metric_specs()
                    config["metrics"] = {
                        name: {"direction": str(spec.direction), "worst": spec.worst_value}
                        for name, spec in specs.items()
                    }
                    if options.ab is not None and specs[user.primary].worst_value is None:
                        raise AgentfileError(
                            f"{user.__name__}.primary {user.primary!r} is lower-is-better, so"
                            " an A/B cannot score a failed execution: declare MetricSpec(name="
                            f"{user.primary!r}, direction='lower', worst=...)",
                            exit_code=2,
                        )
                    tasks = limit(user.tasks, options.limit)
                    loop = _Loop(self, user, tasks, executor, proxy, proxy_url, model_env, store)
                    config["arms"] = _check_arms(loop, options.ab)
                    _write_json(self.run_dir / "config.json", config)
                    self._shape = asyncio.create_task(loop.drive())
                    if self.stopping:
                        self._shape.cancel()
                    try:
                        await self._shape
                    except asyncio.CancelledError:
                        if not self.stopping:
                            raise
        finally:
            config["primary"] = self.primary
            if proxy.token_cap is not None and proxy.tokens_used >= proxy.token_cap:
                config["token_cap_reached"] = {"cap": proxy.token_cap, "used": proxy.tokens_used}
            _write_json(self.run_dir / "config.json", config)
            store.emit(copy.replace(run, config=config, ended_ns=time.time_ns()))
            store.close()
        return RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            statuses=dict(self.statuses),
            primary=self.primary,
            cancelled=self.stopping,
        )


@dataclass(slots=True)
class _Loop:
    runner: Runner
    user_class: type[AgentUser]
    tasks: list[Task]
    executor: ExecutorFn
    proxy: Proxy
    proxy_url: str
    model_env: dict[str, str]
    store: Store
    stage: int = 0
    target: int = 0
    ramping: bool = False
    source: Source | None = None
    alive: set[int] = field(default_factory=set)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    group: asyncio.TaskGroup | None = None
    capped: asyncio.Event = field(default_factory=asyncio.Event)

    async def drive(self) -> None:
        options = self.runner.options
        async with asyncio.TaskGroup() as group:
            self.group = group
            if options.rate_per_s is not None and options.hold_s is not None:
                await self.open_loop(options.rate_per_s, options.hold_s)
                return
            for index, stage in enumerate(options.shape(), start=1):
                if self.capped.is_set():
                    break
                await self.run_stage(index, stage)
            self.set_target(0)

    def check_cap(self) -> None:
        """Once the token cap is reached, start no new execution."""
        cap = self.proxy.token_cap
        if cap is not None and self.proxy.tokens_used >= cap and not self.capped.is_set():
            self.capped.set()
            self.wake()

    def wake(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()

    def set_target(self, users: int) -> None:
        self.target = users
        self.store.emit(
            Load(at_ns=time.time_ns(), stage=self.stage, users=users, ramping=self.ramping)
        )
        assert self.group is not None
        for slot in range(users):
            if slot not in self.alive:
                self.alive.add(slot)
                self.group.create_task(self.user(slot))
        self.wake()

    async def ramp_to(self, users: int) -> None:
        ramp = self.runner.options.ramp_per_min
        if ramp is None or self.target in (0, users):
            self.set_target(users)
            return
        step_s = 60 / ramp
        self.ramping = self.target != users
        while self.target != users:
            self.set_target(self.target + (1 if users > self.target else -1))
            if self.target != users:
                await asyncio.sleep(step_s)
        self.ramping = False

    async def run_stage(self, index: int, stage: Stage) -> None:
        options = self.runner.options
        started_ns = time.time_ns()
        if stage.hold_s is None:
            work = [
                (t, r, arm)
                for r in range(options.repeat)
                for t in self.tasks
                for arm in self.arm_order(t, r)
            ]
            source: Drain | Rounds = Drain(work)
        else:
            source = Rounds(self.tasks, seed=f"{options.seed}:{index}", arms=self.arm_order)
        self.stage, self.source = index, source
        await self.ramp_to(stage.users)
        steady_ns = time.time_ns()
        if isinstance(source, Drain):
            await _either(source.drained.wait(), self.capped.wait())
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.capped.wait(), stage.hold_s or 0)
        self.store.emit(
            StageWindow(
                stage=index,
                users=stage.users,
                hold_s=stage.hold_s,
                started_ns=started_ns,
                steady_ns=steady_ns,
                ended_ns=time.time_ns(),
            )
        )

    async def user(self, slot: int) -> None:
        think_s = self.runner.options.think_time_s
        try:
            while slot < self.target and not self.capped.is_set():
                source = self.source
                item = source.take() if source is not None else None
                if source is None or item is None:
                    await self.changed.wait()
                    continue
                task, repeat, arm = item
                try:
                    await self.execute(task, repeat, arm, self.stage, self.ramping)
                finally:
                    source.finished()
                    self.check_cap()
                if think_s:
                    await asyncio.sleep(think_s)
        finally:
            self.alive.discard(slot)

    async def open_loop(self, rate_per_s: float, hold_s: float) -> None:
        """Tasks arrive on a fixed schedule; an arrival that finds every user busy is shed."""
        options = self.runner.options
        cap = options.max_users or MAX_PROCESS_USERS
        source = Rounds(self.tasks, seed=f"{options.seed}:1", arms=self.arm_order)
        self.stage = 1
        self.store.emit(Load(at_ns=time.time_ns(), stage=1, users=cap, ramping=False))
        running: set[asyncio.Task[None]] = set()
        loop = asyncio.get_running_loop()
        started, started_ns = loop.time(), time.time_ns()
        arrivals = int(hold_s * rate_per_s - 1e-9) + 1
        assert self.group is not None
        for n in range(arrivals):
            if self.capped.is_set():
                break
            await asyncio.sleep(max(0.0, started + n / rate_per_s - loop.time()))
            task, repeat, arm = source.take()
            if len(running) >= cap:
                self.store.emit(Shed(stage=1, at_ns=time.time_ns(), task_id=task.id))
                continue
            execution = self.group.create_task(self.execute(task, repeat, arm, 1, False))
            running.add(execution)
            execution.add_done_callback(running.discard)
            execution.add_done_callback(lambda _: self.check_cap())
        await asyncio.sleep(max(0.0, started + hold_s - loop.time()))
        self.store.emit(
            StageWindow(
                stage=1,
                users=cap,
                hold_s=hold_s,
                started_ns=started_ns,
                steady_ns=started_ns,
                ended_ns=time.time_ns(),
            )
        )

    def arm_order(self, task: Task, repeat: int) -> list[Arm]:
        options = self.runner.options
        return ordered_arms(options.arms(), seed=options.seed, task_id=task.id, repeat=repeat)

    def overlay(self, session: str, arm: Arm) -> dict[str, str]:
        """The variables instar sets for one execution, on top of the inherited environment."""
        env = {var: model_url(self.proxy_url, session, up) for var, up in self.model_env.items()}
        env["OTEL_RESOURCE_ATTRIBUTES"] = _resource_attributes(session)
        return {**env, **arm.env}

    async def execute(self, task: Task, repeat: int, arm: Arm, stage: int, ramping: bool) -> None:
        options = self.runner.options
        session = new_session()
        self.proxy.fault_keys[session] = f"{task.id}:{repeat}"
        env = self.overlay(session, arm)
        started_ns = time.time_ns()
        clock = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="instar-work-") as workdir:
            spec = ExecutionSpec(
                agentfile=options.agentfile,
                task=task,
                session=session,
                arm=arm.name,
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
                self.record(
                    session, task, repeat, arm, stage, ramping, started_ns, clock, cancelled
                )
                raise
        self.record(session, task, repeat, arm, stage, ramping, started_ns, clock, outcome)

    def record(
        self,
        session: str,
        task: Task,
        repeat: int,
        arm: Arm,
        stage: int,
        ramping: bool,
        started_ns: int,
        clock: int,
        outcome: Outcome,
    ) -> None:
        self.runner.statuses[outcome.status] += 1
        self.store.emit(
            Execution(
                session=session,
                task_id=task.id,
                arm=arm.name,
                repeat=repeat,
                stage=stage,
                status=outcome.status,
                started_ns=started_ns,
                ended_ns=started_ns + time.perf_counter_ns() - clock,
                run_ns=outcome.run_ns,
                judge_ns=outcome.judge_ns,
                ramping=ramping,
                error=outcome.error,
                output=outcome.output,
            )
        )
        for name, value in outcome.metrics.items():
            self.store.emit(Metric(session=session, name=name, value=value))
