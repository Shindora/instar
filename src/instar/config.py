from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from instar.proxy.pricing import Price

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|min|h)?\s*$")
RATE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(s|m|min|h)\s*$")
SECONDS: dict[str | None, float] = {"ms": 0.001, "s": 1, "m": 60, "min": 60, "h": 3600, None: 1}


class ConfigError(Exception):
    """Bad usage or configuration; the command exits with code 2."""


def parse_duration(value: object) -> float:
    """`30s`, `5m`, `2h`, `250ms`, or a number of seconds."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    match = DURATION.match(str(value))
    if not match:
        raise ConfigError(f"not a duration: {value!r} (use 30s, 5m or 2h)")
    return float(match[1]) * SECONDS[match[2]]


def parse_rate(value: object) -> float:
    """`2/min`, `120/h` or `5/s`, returned per second."""
    match = RATE.match(str(value))
    if not match:
        raise ConfigError(f"not a rate: {value!r} (use 2/min, 120/h or 5/s)")
    return float(match[1]) / SECONDS[match[2]]


@dataclass(frozen=True, slots=True, kw_only=True)
class Stage:
    """A number of users, ending when every task has run (`hold_s` None) or after `hold_s`."""

    users: int
    hold_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AB:
    """Two arms that differ only by one environment variable: `var` set to `a` or to `b`."""

    var: str
    a: str
    b: str


AB_SPEC = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\s]+)\s*,\s*([^,\s]+)\s*$")


def _ab(value: object) -> AB:
    match = AB_SPEC.match(str(value))
    if not match:
        raise ConfigError(f"--ab takes VARIABLE=a,b with two values: {value!r}")
    if match[2] == match[3]:
        raise ConfigError(f"--ab needs two different values: {value!r}")
    return AB(var=match[1], a=match[2], b=match[3])


def _positive_int(value: object) -> int:
    try:
        number = int(str(value))
    except ValueError:
        raise ConfigError(f"not a whole number: {value!r}") from None
    if number < 1:
        raise ConfigError(f"must be at least 1, got {number}")
    return number


def _positive_float(value: object) -> float:
    try:
        number = float(str(value))
    except ValueError:
        raise ConfigError(f"not a number: {value!r}") from None
    if number <= 0:
        raise ConfigError(f"must be above 0, got {number}")
    return number


def _stage(value: object) -> Stage:
    if isinstance(value, dict):
        unknown = set(value) - {"users", "hold"}
        if unknown or "users" not in value:
            raise ConfigError(f'a stage is a number or {{ users = N, hold = "30m" }}: {value}')
        hold = value.get("hold")
        return Stage(
            users=_positive_int(value["users"]),
            hold_s=None if hold is None else parse_duration(hold),
        )
    return Stage(users=_positive_int(value))


def _stages(value: object) -> tuple[Stage, ...]:
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, list | tuple) or not items:
        raise ConfigError(f"stages must be a non-empty list: {value!r}")
    return tuple(_stage(item) for item in items)


def _faults(value: object) -> dict[str, float]:
    if isinstance(value, str):
        pairs = [item.partition("=") for item in value.split(",") if item.strip()]
        if any(not sep for _, sep, _ in pairs):
            raise ConfigError(f"faults look like http_429=0.1,latency=0.05: {value!r}")
        value = {name.strip(): rate for name, _, rate in pairs}
    if not isinstance(value, dict):
        raise ConfigError(f"faults must map fault names to rates: {value!r}")
    from instar.proxy.faults import check_rates

    rates = {str(name): _rate_value(rate) for name, rate in value.items()}
    try:
        check_rates(rates)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    return rates


def _rate_value(value: object) -> float:
    try:
        return float(str(value))
    except ValueError:
        raise ConfigError(f"fault rate must be a number: {value!r}") from None


def _price_number(model: str, field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ConfigError(
            f"prices.{model}.{field} must be a number of USD per 1M tokens: {value!r}"
        )
    return float(value)


PRICE_FIELDS = ("input", "output", "cache_read", "cache_write", "cache_write_1h")


def _prices(value: object) -> dict[str, Price]:
    """`[tool.instar.prices]`: model (or glob pattern) -> USD per 1M tokens, e.g.

    "claude-sonnet-*" = { input = 3.0, output = 15.0, cache_read = 0.3, cache_write = 3.75 }
    """
    if not isinstance(value, dict):
        raise ConfigError(f"prices must map model ids to tables: {value!r}")
    prices = {}
    for model, table in value.items():
        if not isinstance(table, dict) or not {"input", "output"} <= table.keys():
            raise ConfigError(f"prices.{model} needs input and output (USD per 1M tokens)")
        if unknown := set(table) - set(PRICE_FIELDS):
            raise ConfigError(f"prices.{model} has unknown keys: {', '.join(sorted(unknown))}")
        numbers = {
            name: _price_number(model, name, table[name]) for name in PRICE_FIELDS if name in table
        }
        prices[str(model)] = Price(**numbers)
    return prices


def _strings(value: object) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list | tuple):
        raise ConfigError(f"expected a list of strings: {value!r}")
    return tuple(str(item) for item in items)


def _table(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"expected a table of names to values: {value!r}")
    return {str(k): str(v) for k, v in value.items()}


def _settings(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"[checks] maps check names to settings: {value!r}")
    return {str(k): v for k, v in value.items()}


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"expected true or false: {value!r}")


def _executor(value: object) -> str:
    from instar.executors import executors

    if value not in executors:
        raise ConfigError(f"unknown executor {value!r} (known: {', '.join(executors.names())})")
    return str(value)


OPTIONS: dict[str, tuple[str, Callable[[object], object]]] = {
    "agentfile": ("agentfile", lambda v: Path(str(v))),
    "project": ("project", str),
    "user": ("user", str),
    "executor": ("executor", _executor),
    "users": ("users", _positive_int),
    "stages": ("stages", _stages),
    "hold": ("hold_s", parse_duration),
    "ramp": ("ramp_per_min", lambda v: parse_rate(v) * 60),
    "think-time": ("think_time_s", parse_duration),
    "rate": ("rate_per_s", parse_rate),
    "max-users": ("max_users", _positive_int),
    "repeat": ("repeat", _positive_int),
    "limit": ("limit", _positive_int),
    "timeout-s": ("timeout_s", _positive_float),
    "seed": ("seed", lambda v: int(str(v))),
    "faults": ("faults", _faults),
    "latency-s": ("latency_s", _positive_float),
    "spend-cap": ("spend_cap_usd", _positive_float),
    "token-cap": ("token_cap", _positive_int),
    "prices": ("prices", _prices),
    "keep-outputs": ("keep_outputs", _bool),
    "fail-if": ("fail_if", _strings),
    "baseline": ("baseline", lambda v: Path(str(v))),
    "strict": ("strict", _bool),
    "interval": ("interval_s", parse_duration),
    "breaking-p95": ("breaking_p95", _positive_float),
    "runs-dir": ("runs_dir", lambda v: Path(str(v))),
    "upstreams": ("upstreams", _table),
    "model-env": ("model_env", _table),
    "ab": ("ab", _ab),
    "telemetry-map": ("telemetry_map", lambda v: Path(str(v))),
    "checks": ("checks", _settings),
    "span-grace": ("span_grace_s", parse_duration),
    "otel-export": ("otel_export", str),
}
SHAPE_FLAGS = frozenset({"users", "hold"})


def load_project(root: Path) -> dict[str, object]:
    """`[tool.instar]` from pyproject.toml, or the same tables at the top of instar.toml."""
    own = root / "instar.toml"
    if own.exists():
        return tomllib.loads(own.read_text())
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        return tomllib.loads(pyproject.read_text()).get("tool", {}).get("instar", {})
    return {}


def resolve(
    flags: Mapping[str, object], *, profile: str | None, project: Mapping[str, object]
) -> dict[str, object]:
    """Project defaults, then the profile, then command-line flags, each converted once."""
    defaults = {k: v for k, v in project.items() if k != "profiles"}
    profiles = project.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ConfigError("profiles must be tables")
    if profile is not None and profile not in profiles:
        known = ", ".join(sorted(profiles)) or "none"
        raise ConfigError(f"no profile {profile!r} (known: {known})")
    raw = {**defaults, **(profiles[profile] if profile else {}), **flags}
    if SHAPE_FLAGS & flags.keys() and "stages" not in flags:
        raw.pop("stages", None)
    if unknown := sorted(set(raw) - OPTIONS.keys()):
        raise ConfigError(f"unknown option {', '.join(unknown)}")
    resolved = {OPTIONS[key][0]: OPTIONS[key][1](value) for key, value in raw.items()}
    if resolved.get("rate_per_s") is not None:
        if resolved.get("hold_s") is None:
            raise ConfigError("--rate needs --hold: how long tasks keep arriving")
        if resolved.get("stages"):
            raise ConfigError("--rate replaces stages; give one or the other")
    if resolved.get("ab") is not None and resolved.get("executor") == "inline":
        raise ConfigError("--ab needs an executor with per-execution environment; inline has none")
    if resolved.get("stages") and resolved.get("hold_s") is not None:
        raise ConfigError("--hold applies to one stage; with stages, give each its own hold")
    return resolved
