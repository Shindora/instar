from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from instar.config import ConfigError
from instar.events import SpanKind
from instar.telemetry import semconv

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

COST_UNITS: dict[str, float] = {"usd": 1.0, "usd_cents": 0.01, "usd_micros": 1e-6}
SECTIONS: dict[str, frozenset[str]] = {
    "kinds": frozenset({"operation", "llm", "tool", "agent"}),
    "names": frozenset({"tool", "tool_arguments"}),
    "cost": frozenset({"attribute", "unit"}),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class TelemetryMap:
    """Which span attributes mean what. Defaults follow the GenAI conventions; a project file
    overrides them and is the only place a cost attribute can come from:

        [kinds]
        operation = "gen_ai.operation.name"
        llm = ["chat"]
        [names]
        tool = "gen_ai.tool.name"
        [cost]
        attribute = "llm.usage.cost"
        unit = "usd"            # usd, usd_cents or usd_micros
    """

    operation: str = semconv.OPERATION
    llm: frozenset[str] = semconv.LLM_OPERATIONS
    tool: frozenset[str] = semconv.TOOL_OPERATIONS
    agent: frozenset[str] = semconv.AGENT_OPERATIONS
    tool_name: str = semconv.TOOL_NAME
    tool_arguments: str = semconv.TOOL_ARGUMENTS
    cost_attribute: str | None = None
    cost_unit: str = "usd"
    kept: frozenset[str] = field(default=semconv.KEPT_ATTRIBUTES)

    @classmethod
    def load(cls, path: Path | None) -> TelemetryMap:
        if path is None:
            return cls()
        try:
            data = tomllib.loads(path.read_text())
        except FileNotFoundError:
            raise ConfigError(f"telemetry map not found: {path}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"telemetry map {path} is not TOML: {exc}") from None
        return cls.from_tables(data, where=str(path))

    @classmethod
    def from_tables(
        cls, data: Mapping[str, object], *, where: str = "telemetry map"
    ) -> TelemetryMap:
        for section, value in data.items():
            if section not in SECTIONS or not isinstance(value, dict):
                raise ConfigError(f"{where}: unknown section [{section}]")
            if unknown := set(value) - SECTIONS[section]:
                raise ConfigError(
                    f"{where}: unknown key {', '.join(sorted(unknown))} in [{section}]"
                )
        kinds = _table(data, "kinds")
        names = _table(data, "names")
        cost = _table(data, "cost")
        unit = str(cost.get("unit", "usd"))
        if unit not in COST_UNITS:
            raise ConfigError(f"{where}: cost unit must be one of {', '.join(COST_UNITS)}")
        if "unit" in cost and "attribute" not in cost:
            raise ConfigError(f"{where}: [cost] names a unit but no attribute")
        default = cls()
        mapped = cls(
            operation=str(kinds.get("operation", default.operation)),
            llm=_names(kinds.get("llm"), default.llm),
            tool=_names(kinds.get("tool"), default.tool),
            agent=_names(kinds.get("agent"), default.agent),
            tool_name=str(names.get("tool", default.tool_name)),
            tool_arguments=str(names.get("tool_arguments", default.tool_arguments)),
            cost_attribute=str(cost["attribute"]) if "attribute" in cost else None,
            cost_unit=unit,
        )
        extra = {mapped.operation, mapped.tool_name, mapped.cost_attribute} - {None}
        return copy.replace(mapped, kept=semconv.KEPT_ATTRIBUTES | {str(n) for n in extra})

    def to_tables(self) -> dict[str, object]:
        tables: dict[str, object] = {
            "kinds": {
                "operation": self.operation,
                "llm": sorted(self.llm),
                "tool": sorted(self.tool),
                "agent": sorted(self.agent),
            },
            "names": {"tool": self.tool_name, "tool_arguments": self.tool_arguments},
        }
        if self.cost_attribute is not None:
            tables["cost"] = {"attribute": self.cost_attribute, "unit": self.cost_unit}
        return tables

    def kind(self, attributes: Mapping[str, object]) -> SpanKind:
        """By the operation attribute; without one, a tool name means tool and a model, llm."""
        operation = attributes.get(self.operation)
        if operation in self.llm:
            return SpanKind.LLM
        if operation in self.tool:
            return SpanKind.TOOL
        if operation in self.agent:
            return SpanKind.AGENT
        if operation is None and self.tool_name in attributes:
            return SpanKind.TOOL
        if operation is None and semconv.REQUEST_MODEL in attributes:
            return SpanKind.LLM
        return SpanKind.OTHER

    def keep(self, attributes: Mapping[str, object]) -> dict[str, object]:
        """Only an allowlist: span attributes may carry prompts, replies and tool arguments,
        and none of those is ever stored. Tool arguments survive only as a hash."""
        return {k: v for k, v in attributes.items() if k in self.kept and k != self.tool_arguments}

    def cost_usd(self, attributes: Mapping[str, object]) -> float | None:
        if self.cost_attribute is None:
            return None
        value = attributes.get(self.cost_attribute)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value) * COST_UNITS[self.cost_unit]


def _table(data: Mapping[str, object], name: str) -> dict[str, object]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _names(value: object, default: frozenset[str]) -> frozenset[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, list):
        return frozenset(str(v) for v in value)
    raise ConfigError(f"expected a name or a list of names: {value!r}")
