from __future__ import annotations

from dataclasses import dataclass

import pytest

from instar.judge import as_metrics
from instar.registry import Registry


@dataclass
class Score:
    passed: bool
    similarity: float


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (True, {"passed": True}),
        (False, {"passed": False}),
        (3, {"score": 3.0}),
        (0.5, {"score": 0.5}),
        ({"passed": True, 1: "x"}, {"passed": True, "1": "x"}),
        (Score(passed=True, similarity=0.9), {"passed": True, "similarity": 0.9}),
    ],
)
def test_as_metrics_accepts_judge_results(result: object, expected: dict[str, object]) -> None:
    assert as_metrics(result) == expected


def test_as_metrics_keeps_bool_as_bool() -> None:
    assert as_metrics(True)["passed"] is True


@pytest.mark.parametrize("result", ["yes", None, Score])
def test_as_metrics_rejects_unsupported(result: object) -> None:
    with pytest.raises(TypeError, match="unsupported type"):
        as_metrics(result)


def test_registry_registers_and_looks_up() -> None:
    faults: Registry[int] = Registry("fault")
    faults("b")(2)
    faults("a")(1)
    assert faults["a"] == 1
    assert "b" in faults
    assert faults.names() == ["a", "b"]


def test_registry_rejects_duplicates_and_names_known_entries() -> None:
    checks: Registry[int] = Registry("check")
    checks("max_steps")(1)
    with pytest.raises(ValueError, match="duplicate check: 'max_steps'"):
        checks("max_steps")(2)
    with pytest.raises(KeyError, match="known: max_steps"):
        checks["missing"]
