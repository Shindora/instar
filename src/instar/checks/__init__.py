"""Behaviour checks: verdicts on one execution from its model calls and, when the agent emits
telemetry, its spans. A check without the data it needs reports "not evaluated", never a pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from instar.events import CheckResult, CheckStatus, SpanKind
from instar.registry import Registry
from instar.telemetry import semconv

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from instar.events import Execution, LLMCall, Span

CALLS = "calls"
SPANS = "spans"


@dataclass(frozen=True, slots=True)
class Verdict:
    status: CheckStatus
    detail: str


def passed(detail: str) -> Verdict:
    return Verdict(CheckStatus.PASSED, detail)


def failed(detail: str) -> Verdict:
    return Verdict(CheckStatus.FAILED, detail)


def not_evaluated(detail: str) -> Verdict:
    return Verdict(CheckStatus.NOT_EVALUATED, detail)


@dataclass(frozen=True, slots=True, kw_only=True)
class Evidence:
    """What one execution left behind: its agent model calls and its agent spans, both in
    order. Judge calls and judge spans are never here."""

    execution: Execution
    calls: Sequence[LLMCall]
    spans: Sequence[Span]
    tool_name_attribute: str = semconv.TOOL_NAME

    def tools(self) -> list[tuple[str, str | None]]:
        """(tool name, hash of its arguments) for each tool span, in start order."""
        return [
            (
                str(span.attributes.get(self.tool_name_attribute) or span.name),
                _text(span.attributes.get(semconv.TOOL_ARGUMENTS_HASH)),
            )
            for span in self.spans
            if span.kind is SpanKind.TOOL
        ]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


type CheckFn = Callable[[Evidence, Any], Verdict]


@dataclass(frozen=True, slots=True, kw_only=True)
class Check:
    name: str
    needs: frozenset[str]
    default: object
    fn: CheckFn


checks: Registry[Check] = Registry("check")


def check(
    name: str, *, needs: Sequence[str] = (CALLS,), default: object = True
) -> Callable[[CheckFn], CheckFn]:
    """Register a behaviour check: `fn(evidence, setting) -> Verdict`.

    `needs` names the data it reads ("calls", "spans"). `default` is the setting it runs with
    when the project configures nothing; None means it runs only when configured.
    """

    def decorator(fn: CheckFn) -> CheckFn:
        checks(name)(Check(name=name, needs=frozenset(needs), default=default, fn=fn))
        return fn

    return decorator


def enabled(configured: Mapping[str, object]) -> dict[str, object]:
    """Built-in defaults, then the project's `[checks]`: a value sets a check, `false` turns it
    off, `true` uses its default. Raises KeyError for an unknown name."""
    settings = {
        name: checks[name].default for name in checks.names() if checks[name].default is not None
    }
    for name, value in configured.items():
        spec = checks[name]
        if value is False:
            settings.pop(name, None)
        elif value is True:
            if spec.default is None:
                raise ValueError(f"check {name!r} has no default; give it a setting")
            settings[name] = spec.default
        else:
            settings[name] = value
    return settings


def evaluate(evidence: Evidence, settings: Mapping[str, object]) -> list[CheckResult]:
    results = []
    for name, setting in settings.items():
        spec = checks[name]
        if SPANS in spec.needs and not evidence.spans:
            verdict = not_evaluated("no spans arrived for this execution")
        else:
            try:
                verdict = spec.fn(evidence, setting)
            except Exception as exc:  # a project's check must not end the run
                verdict = not_evaluated(f"the check raised {exc!r}")
        results.append(
            CheckResult(
                session=evidence.execution.session,
                check=name,
                status=verdict.status,
                detail=verdict.detail,
            )
        )
    return results


from instar.checks import builtin as builtin  # noqa: E402  registers the built-in checks
