from __future__ import annotations

import hashlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from instar.dataset import Task
from instar.proxy import upstreams

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class Context:
    """What `run` may need besides the task. Most agentfiles never touch it."""

    session: str
    arm: str
    workdir: Path
    proxy_url: str

    def model_url(self, upstream: str) -> str:
        """Base URL for the agent's model client, carrying this execution's session."""
        return upstreams.model_url(self.proxy_url, self.session, upstream)

    def judge_url(self, upstream: str) -> str:
        """Base URL for model calls made by the judge: logged, never faulted, never capped."""
        return upstreams.judge_url(self.proxy_url, self.session, upstream)


class AgentUser:
    """Subclass this in an agentfile: set `tasks`, write `run`, optionally write `judge`.

    `run` and `judge` may each be `def` or `async def`. `judge` may return a bool, a number, a
    mapping or a dataclass; `primary` names the metric that means "resolved".
    """

    registry: ClassVar[dict[str, type[AgentUser]]] = {}
    tasks: ClassVar[Sequence[Task]] = ()
    primary: ClassVar[str] = "passed"

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not abstract:
            AgentUser.registry[f"{cls.__module__}:{cls.__qualname__}"] = cls

    def run(self, task: Task, ctx: Context) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must define run(task, ctx)")

    def judge(self, task: Task, output: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def has_judge(cls) -> bool:
        return cls.judge is not AgentUser.judge


class AgentfileError(Exception):
    """An agentfile could not be used. `exit_code` is 3 when it failed to import, else 2."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _import(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode(), usedforsecurity=False).hexdigest()[:8]
    name = f"instar_agentfile_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AgentfileError(f"cannot import agentfile {path}")
    for entry in (str(Path.cwd()), str(path.parent)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise AgentfileError(
            f"agentfile {path} failed to import:\n{traceback.format_exc()}", exit_code=3
        ) from None
    return name


def load_agentfile(path: Path, user: str | None = None) -> type[AgentUser]:
    """Import an agentfile and return its AgentUser subclass, checking its tasks."""
    path = path.resolve()
    if not path.is_file():
        raise AgentfileError(f"agentfile not found: {path}")
    module = _import(path)
    found = {c.__name__: c for c in AgentUser.registry.values() if c.__module__ == module}
    if not found:
        raise AgentfileError(f"{path} defines no AgentUser subclass")
    if user is None and len(found) > 1:
        names = ", ".join(sorted(found))
        raise AgentfileError(f"{path} defines several AgentUser classes ({names}); pick one")
    if user is not None and user not in found:
        raise AgentfileError(f"{path} has no AgentUser class {user!r}")
    cls = found[user] if user is not None else next(iter(found.values()))
    if not cls.tasks:
        raise AgentfileError(f"{cls.__name__}.tasks is empty")
    if not all(isinstance(task, Task) for task in cls.tasks):
        raise AgentfileError(f"{cls.__name__}.tasks must hold instar.Task objects")
    ids = [task.id for task in cls.tasks]
    if len(set(ids)) != len(ids):
        raise AgentfileError(f"{cls.__name__}.tasks has duplicate task ids")
    return cls
