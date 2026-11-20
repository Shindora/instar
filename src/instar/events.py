from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class CallRole(StrEnum):
    AGENT = "agent"
    JUDGE = "judge"


class ExecutionStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CRASHED = "crashed"


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCall:
    """One model call seen by the proxy. Bodies and auth headers are never kept."""

    session: str
    role: CallRole
    upstream: str
    index: int
    method: str
    path: str
    model: str | None
    status: int
    started_ns: int
    ended_ns: int
    first_byte_ns: int | None = None
    fault: str | None = None
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Execution:
    """One attempt at one task, identified by (task_id, arm, repeat) within a run."""

    session: str
    task_id: str
    arm: str
    repeat: int
    stage: int
    status: ExecutionStatus
    started_ns: int
    run_ns: int
    judge_ns: int | None = None
    ramping: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    """One value a judge or an importer reported for one execution."""

    session: str
    name: str
    value: bool | int | float | str | None
