from __future__ import annotations

import re
from pathlib import Path

import pytest

from instar.analyze.gate import Gate, GateState, evaluate
from instar.analyze.stats import Estimate
from instar.cli import main
from instar.config import ConfigError, Stage, load_project, parse_duration, parse_rate, resolve
from instar.events import Execution
from instar.store import StoreReader

LOAD = Path(__file__).parents[1] / "tests" / "fixtures" / "agentfile_load.py"

PROJECT = {
    "agentfile": "evals/agentfile.py",
    "repeat": 2,
    "profiles": {
        "smoke": {"limit": 10, "users": 4, "fail-if": ["passed < 0.7"]},
        "release": {"stages": [1, 4, 8], "faults": {"http_429": 0.1}},
        "peak": {
            "ramp": "2/min",
            "think-time": "10s",
            "spend-cap": 50,
            "stages": [{"users": 5, "hold": "5m"}, {"users": 30, "hold": "30m"}],
        },
    },
}


@pytest.mark.parametrize(
    ("text", "seconds"), [("30s", 30), ("5m", 300), ("2h", 7200), ("250ms", 0.25), (90, 90)]
)
def test_durations(text: object, seconds: float) -> None:
    assert parse_duration(text) == seconds


def test_rates() -> None:
    assert parse_rate("120/h") == pytest.approx(1 / 30)
    assert parse_rate("2/min") == pytest.approx(1 / 30)
    with pytest.raises(ConfigError, match="not a rate"):
        parse_rate("fast")


def test_profiles_resolve_with_flags_overriding_them() -> None:
    smoke = resolve({}, profile="smoke", project=PROJECT)
    assert smoke["limit"] == 10
    assert smoke["repeat"] == 2
    assert smoke["fail_if"] == ("passed < 0.7",)
    assert resolve({"limit": "3"}, profile="smoke", project=PROJECT)["limit"] == 3

    peak = resolve({}, profile="peak", project=PROJECT)
    assert peak["stages"] == (Stage(users=5, hold_s=300), Stage(users=30, hold_s=1800))
    assert peak["ramp_per_min"] == pytest.approx(2)
    assert peak["think_time_s"] == 10
    assert peak["spend_cap_usd"] == 50

    single = resolve({"users": "8", "hold": "1m"}, profile="release", project=PROJECT)
    assert "stages" not in single
    assert (single["users"], single["hold_s"]) == (8, 60)
    assert resolve({}, profile="release", project=PROJECT)["faults"] == {"http_429": 0.1}


@pytest.mark.parametrize(
    ("flags", "profile", "message"),
    [
        ({}, "missing", "no profile 'missing' (known: peak, release, smoke)"),
        ({"rate": "120/h"}, None, "--rate needs --hold"),
        ({"stages": "1,2", "hold": "1m"}, None, "--hold applies to one stage"),
        ({"faults": "nope=0.1"}, None, "unknown fault 'nope'"),
        ({"users": "0"}, None, "must be at least 1"),
        ({"colour": "red"}, None, "unknown option colour"),
    ],
)
def test_bad_configuration_is_reported(
    flags: dict[str, object], profile: str | None, message: str
) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        resolve(flags, profile=profile, project=PROJECT)


def test_project_config_comes_from_instar_toml_or_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.instar]\nproject = "from-pyproject"\n')
    assert load_project(tmp_path) == {"project": "from-pyproject"}
    (tmp_path / "instar.toml").write_text('project = "from-instar-toml"\n')
    assert load_project(tmp_path) == {"project": "from-instar-toml"}


@pytest.mark.usefixtures("model_env")
def test_cli_runs_a_profile_from_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.instar]\n"
        f'agentfile = "{LOAD}"\n'
        'project = "demo"\n'
        'executor = "inline"\n'
        "[tool.instar.profiles.smoke]\n"
        "limit = 3\n"
        'fail-if = ["passed < 0.1"]\n'
    )
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--profile", "smoke"]) == 0
    assert main(["run", "--profile", "smoke", "--limit", "1"]) == 0
    first, second = sorted((tmp_path / "runs" / "demo").iterdir())
    assert len(StoreReader(first).select(Execution)) == 3
    assert len(StoreReader(second).select(Execution)) == 1


def test_gate_sentences_and_inconclusive_state() -> None:
    estimate = Estimate(value=0.86, low=0.79, high=0.91, n=100)
    [inside] = evaluate(
        [Gate.parse("passed < 0.83")],
        {"passed": estimate},
        baseline=None,
        percent_metrics=frozenset({"passed"}),
    )
    assert inside.state is GateState.INCONCLUSIVE
    assert inside.sentence.startswith("Fails below 83%. Now 86%, likely between 79% and 91%.")
    [relative] = evaluate(
        [Gate.parse("cost_per_pass > baseline * 1.2")],
        {"cost_per_pass": Estimate(value=0.5)},
        baseline={"cost_per_pass": 0.4},
        percent_metrics=frozenset(),
    )
    assert relative.threshold == pytest.approx(0.48)
    assert relative.state is GateState.FAILED
    assert relative.sentence == "Fails above 0.48. Now 0.5. Failed."
