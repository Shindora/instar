from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from instar.analyze.gate import Gate, GateState, evaluate
from instar.analyze.stats import Estimate
from instar.cli import main
from instar.config import ConfigError, Stage, load_project, parse_duration, parse_rate, resolve
from instar.events import Execution, LLMCall
from instar.proxy.pricing import Price
from instar.store import StoreReader

if TYPE_CHECKING:
    from tests.fixtures.mock_upstream import MockUpstream

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


def test_prices_and_token_cap_resolve() -> None:
    project = {
        "prices": {
            "gpt-*": {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75},
            "y": {"input": 1, "output": 2},
        }
    }
    resolved = resolve({"token-cap": "5000"}, profile=None, project=project)
    assert resolved["prices"] == {
        "gpt-*": Price(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
        "y": Price(input=1.0, output=2.0),
    }
    assert resolved["token_cap"] == 5000
    with pytest.raises(ConfigError, match=r"prices\.z needs input and output"):
        resolve({}, profile=None, project={"prices": {"z": {"input": 1}}})
    with pytest.raises(ConfigError, match="unknown keys: cached"):
        resolve({}, profile=None, project={"prices": {"z": {"input": 1, "output": 1, "cached": 1}}})
    with pytest.raises(ConfigError, match="unknown keys: currency"):
        resolve(
            {},
            profile=None,
            project={"prices": {"z": {"input": 1, "output": 1, "currency": "eur"}}},
        )


@pytest.mark.usefixtures("model_env")
def test_price_table_gives_cost_per_pass_and_a_token_cap_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "instar.toml").write_text(
        f'agentfile = "{LOAD}"\nproject = "priced"\nexecutor = "inline"\nusers = 1\n'
        '[prices."mock-model"]\ninput = 1.0\noutput = 2.0\n'
    )
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--limit", "4", "--fail-if", "cost_per_pass > 0.001"]) == 0
    assert main(["run", "--limit", "6", "--token-cap", "12"]) == 3
    priced, capped = sorted((tmp_path / "runs" / "priced").iterdir())
    report = json.loads((priced / "report.json").read_text())
    assert report["metrics"]["cost_per_pass"]["value"] == pytest.approx(8e-6)
    assert report["gates"][0]["state"] == "passed"
    assert report["cost_note"] is None
    capped_report = json.loads((capped / "report.json").read_text())
    assert capped_report["summary"]["statuses"] == {"ok": 3}
    assert capped_report["token_cap"] == {"cap": 12, "used": 15}
    assert "(exceeded by 3); no execution started after that" in (capped / "report.md").read_text()
    assert [c.fault for c in StoreReader(capped).select(LLMCall)] == [None, None, None]


@pytest.mark.usefixtures("model_env")
def test_unpriced_models_make_cost_incomplete_and_missing_usage_a_lower_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_upstream: MockUpstream,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = f'agentfile = "{LOAD}"\nproject = "costs"\nexecutor = "inline"\n'
    (tmp_path / "instar.toml").write_text(f'{config}[prices."gpt-*"]\ninput = 1.0\noutput = 2.0\n')
    monkeypatch.chdir(tmp_path)
    gate = ["--fail-if", "cost_per_pass > 0.001"]
    assert main(["run", "--limit", "2", *gate, "--strict"]) == 1
    out = capsys.readouterr().out
    assert "unpriced models: mock-model; total cost is incomplete" in out
    assert "gate not_evaluated: cost_per_pass > 0.001" in out

    (tmp_path / "instar.toml").write_text(f'{config}[prices."mock-*"]\ninput = 1.0\noutput = 2.0\n')
    mock_upstream.omit_usage = True
    assert main(["run", "--limit", "2", *gate]) == 0
    out = capsys.readouterr().out
    assert "2 model calls ended without token usage; spend is a lower bound" in out
    assert "Now at least 0" in out
