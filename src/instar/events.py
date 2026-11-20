from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class CallRole(StrEnum):
    AGENT = "agent"
    JUDGE = "judge"


class ExecutionStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    CANCELLED = "cancelled"


class SpanKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    OTHER = "other"


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    """One invocation of `instar run`: written at its start, then again with `ended_ns`."""

    run_id: str
    project: str
    started_ns: int
    ended_ns: int | None = None
    config: dict[str, object] = field(default_factory=dict)


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
    traceparent: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    output_tokens: int = 0
    response_model: str | None = None
    usage_known: bool = True
    cost_usd: float | None = None
    cost_note: str | None = None
    cost_approximate: bool = False


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
    ended_ns: int
    run_ns: int
    judge_ns: int | None = None
    ramping: bool = False
    error: str | None = None
    output: object = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    """One value a judge or an importer reported for one execution."""

    session: str
    name: str
    value: bool | int | float | str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Span:
    """One span from the agent's own telemetry, normalised and joined to its execution."""

    session: str
    trace_id: str
    span_id: str
    parent_id: str | None
    kind: SpanKind
    name: str
    started_ns: int
    ended_ns: int
    error: bool = False
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckResult:
    """The verdict of one behaviour check on one execution."""

    session: str
    check: str
    status: CheckStatus
    detail: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Load:
    """The target number of users changed: at a stage start, or one ramp step."""

    at_ns: int
    stage: int
    users: int
    ramping: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class StageWindow:
    """When one stage ran. `steady_ns` is when its ramp finished; timed stages have `hold_s`."""

    stage: int
    users: int
    hold_s: float | None
    started_ns: int
    steady_ns: int
    ended_ns: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Shed:
    """An open-loop arrival that found every user busy, so it never ran."""

    stage: int
    at_ns: int
    task_id: str


type Event = Run | LLMCall | Execution | Metric | Span | CheckResult | Load | StageWindow | Shed


class Sink(Protocol):
    """Where producers send events. `emit` must not block the event loop for long."""

    def emit(self, event: Event) -> None: ...
