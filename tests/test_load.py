from __future__ import annotations

import asyncio
import itertools
import json
import socket
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from instar.analyze import analyze
from instar.cli import main
from instar.config import Stage
from instar.events import Execution, Load, StageWindow
from instar.runner import Runner, RunOptions
from instar.store import StoreReader

if TYPE_CHECKING:
    from tests.fixtures.mock_upstream import MockUpstream

ROOT = Path(__file__).parents[1]
LOAD = ROOT / "tests" / "fixtures" / "agentfile_load.py"

pytestmark = pytest.mark.usefixtures("model_env")


def cli(tmp_path: Path, *args: str) -> tuple[int, Path | None]:
    runs = tmp_path / "runs"
    code = main(["run", "-f", str(LOAD), "--executor", "inline", "--runs-dir", str(runs), *args])
    project = runs / Path.cwd().name
    dirs = sorted(project.iterdir()) if project.exists() else []
    return code, dirs[-1] if dirs else None


def run_options(tmp_path: Path, **options: object) -> RunOptions:
    base = RunOptions(agentfile=LOAD, project="load", runs_dir=tmp_path, executor="inline")
    return RunOptions(**{**vars_of(base), **options})  # ty: ignore[invalid-argument-type]  overrides by field name


def vars_of(options: RunOptions) -> dict[str, object]:
    return {name: getattr(options, name) for name in RunOptions.__dataclass_fields__}


def in_flight(executions: list[Execution], at_ns: int) -> int:
    return sum(1 for e in executions if e.started_ns <= at_ns < e.ended_ns)


def test_breaking_point_is_the_stage_where_the_agent_degrades(
    tmp_path: Path, mock_upstream: MockUpstream
) -> None:
    mock_upstream.delay_s, mock_upstream.capacity = 0.1, 8
    code, run_dir = cli(tmp_path, "--stages", "2,4,8,16", "--breaking-p95", "1000")
    assert code == 0
    assert run_dir is not None
    report = (run_dir / "report.md").read_text()
    assert "**Breaking point: stage 4 (16 users).** primary 50% fell below stage 1's" in report
    data = json.loads((run_dir / "report.json").read_text())
    assert [s["steady"]["executions"] for s in data["stages"]] == [16, 16, 16, 16]
    assert data["breaking_point"]["stage"] == 4


def test_timed_stage_holds_its_users_and_ends_on_time(
    tmp_path: Path, mock_upstream: MockUpstream
) -> None:
    mock_upstream.delay_s = 0.1
    result = asyncio.run(Runner(run_options(tmp_path, users=4, hold_s=1.5)).run())
    reader = StoreReader(result.run_dir)
    [window] = reader.select(StageWindow)
    executions = reader.select(Execution)
    assert 1.5 <= (window.ended_ns - window.steady_ns) / 1e9 < 2.5
    assert max(e.started_ns for e in executions) <= window.ended_ns
    steps = range(window.steady_ns + 150_000_000, window.ended_ns - 50_000_000, 25_000_000)
    counts = [in_flight(executions, t) for t in steps]
    assert max(counts) <= 4
    assert sum(3 <= c <= 4 for c in counts) >= 0.9 * len(counts), counts
    assert max(e.ended_ns for e in executions) - window.ended_ns < 2_000_000_000


def test_ramp_adds_and_removes_users_at_the_configured_rate(
    tmp_path: Path, mock_upstream: MockUpstream
) -> None:
    mock_upstream.delay_s = 0.05
    stages = (Stage(users=1, hold_s=0.5), Stage(users=4, hold_s=0.5), Stage(users=1, hold_s=0.3))
    result = asyncio.run(Runner(run_options(tmp_path, stages=stages, ramp_per_min=300)).run())
    reader = StoreReader(result.run_dir)
    loads = reader.select(Load)
    up = [load for load in loads if load.stage == 2]
    down = [load for load in loads if load.stage == 3]
    assert [load.users for load in up] == [2, 3, 4]
    down = [load for load in down if load.users]
    assert [load.users for load in down] == [3, 2, 1]
    for steps in (up, down):
        gaps = [(b.at_ns - a.at_ns) / 1e9 for a, b in itertools.pairwise(steps)]
        assert all(0.19 <= gap < 1.0 for gap in gaps), gaps
    windows = {w.stage: w for w in reader.select(StageWindow)}
    assert 0.39 <= (windows[2].steady_ns - windows[2].started_ns) / 1e9 < 1.5
    executions = reader.select(Execution)
    assert any(e.ramping for e in executions if e.stage == 2)
    assert all(not e.ramping for e in executions if e.stage == 1)


def test_open_loop_over_its_user_cap_reports_shed_arrivals(
    tmp_path: Path, mock_upstream: MockUpstream
) -> None:
    mock_upstream.delay_s = 0.3
    code, run_dir = cli(tmp_path, "--rate", "20/s", "--hold", "1s", "--max-users", "2")
    assert code == 0
    assert run_dir is not None
    data = json.loads((run_dir / "report.json").read_text())
    [stage] = data["stages"]
    assert stage["arrivals"] == 20
    assert 0 < stage["shed"] < 20
    assert stage["steady"]["executions"] + stage["shed"] == 20
    assert data["metrics"]["shed_rate"]["value"] == pytest.approx(stage["shed"] / 20)
    assert "| 1 | 2 | 1 s |" in (run_dir / "report.md").read_text()


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (["--fail-if", "passed < 0.5"], 0),
        (["--fail-if", "passed > 0.5"], 1),
        (["--limit", "3", "--fail-if", "passed < 0.6"], 0),
        (["--limit", "3", "--fail-if", "passed < 0.6", "--strict"], 1),
        (["--fail-if", "passed <<< 1"], 2),
        (["--fail-if", "nope < 1"], 2),
        (["--fail-if", "passed < baseline"], 2),
        (["--profile", "missing"], 2),
        (["--fail-if", "cost_per_pass > 1"], 0),
        (["--fail-if", "cost_per_pass > 1", "--strict"], 1),
        (["--fail-if", "tasks_per_hour < 1"], 0),
    ],
)
def test_gate_exit_codes(tmp_path: Path, args: list[str], code: int) -> None:
    assert cli(tmp_path, *args)[0] == code


def test_unreachable_upstream_means_the_test_could_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed = sock.getsockname()[1]
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{closed}/v1")
    assert cli(tmp_path, "--limit", "2")[0] == 3
    assert "every model call failed (HTTP 502 x2)" in capsys.readouterr().err


def test_baseline_comparison_reports_per_task_regressions(tmp_path: Path) -> None:
    code, first = cli(tmp_path, "--limit", "8")
    assert code == 0
    assert first is not None
    baseline = tmp_path / "baseline.json"
    assert main(["baseline", "save", str(first), str(baseline)]) == 0
    saved = json.loads(baseline.read_text())
    assert saved["summary"]["passed"] == 1.0
    assert len(saved["tasks"]) == 8

    args = ["--limit", "8", "--faults", "http_500=0.5", "--seed", "3", "--baseline", str(baseline)]
    code, second = cli(tmp_path, *args, "--fail-if", "passed < baseline - 0.05")
    assert code == 1
    assert second is not None
    report = (second / "report.md").read_text()
    data = json.loads((second / "report.json").read_text())
    regressed = [c["task_id"] for c in data["comparison"]["changes"] if c["regressed"]]
    assert 0 < len(regressed) < 8
    assert f"{len(regressed)} tasks regressed:" in report
    assert all(f"- `{task}`: 100% -> 0%" in report for task in regressed)


def test_baseline_records_its_source_and_compares_shared_tasks_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, first = cli(tmp_path, "--limit", "8", "--seed", "5")
    assert code == 0
    assert first is not None
    path = tmp_path / "baseline.json"
    assert main(["baseline", "save", str(first), str(path)]) == 0
    saved = json.loads(path.read_text())
    assert saved["run_id"] == first.name
    assert saved["seed"] == 5
    assert saved["models"] == ["mock-model"]
    assert saved["evaluation_version"]
    assert saved["date"].endswith("+00:00")
    assert saved["task_ids"] == sorted(saved["task_ids"]) == [f"t{i:02d}" for i in range(8)]

    saved["models"], saved["evaluation_version"] = ["other-model"], "0000older"
    path.write_text(json.dumps(saved))
    capsys.readouterr()
    code, second = cli(
        tmp_path, "--limit", "10", "--baseline", str(path), "--fail-if", "passed < baseline - 0.1"
    )
    assert code == 0
    assert second is not None
    comparison = json.loads((second / "report.json").read_text())["comparison"]
    assert len(comparison["changes"]) == 8
    assert comparison["only_now"] == ["t08", "t09"]
    assert comparison["only_baseline"] == []
    err = capsys.readouterr().err
    assert "model ids differ: baseline other-model, now mock-model" in err
    assert "evaluation code differs: baseline 0000older" in err
    report = (second / "report.md").read_text()
    assert "8 tasks in both runs; unmatched: 2 only in this run, 0 only in the baseline." in report


def test_junit_has_one_case_per_task_and_repeats_are_reported_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code, run_dir = cli(tmp_path, "--limit", "5", "--repeat", "2", "--fail-if", "passed < 0.2")
    assert code == 0
    assert run_dir is not None
    suites = ET.parse(run_dir / "junit.xml").getroot()
    cases = suites.findall("testsuite")[0].findall("testcase")
    assert sorted(str(c.get("name")) for c in cases) == [f"t{i:02d}" for i in range(5)]
    assert [c.get("name") for c in suites.findall("testsuite")[1]] == ["passed < 0.2"]
    report = (run_dir / "report.md").read_text()
    assert "| First attempts | 5 |" in report
    assert "| Repeats | 5 |" in report
    assert summary.read_text() == report


def test_time_series_counts_every_execution(tmp_path: Path) -> None:
    result = asyncio.run(Runner(run_options(tmp_path, limit=6, interval_s=0.05)).run())
    analysis = analyze(result.run_dir, run_options(tmp_path, interval_s=0.05))
    assert sum(b.started for b in analysis.series) == 6
    assert sum(b.finished for b in analysis.series) == 6
    assert analysis.series[0].users == 4


def test_timed_run_prints_a_projection_from_the_last_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = str(tmp_path / "runs")
    assert main(["doctor", "-f", str(LOAD), "--executor", "inline", "--runs-dir", runs]) == 0
    capsys.readouterr()
    assert cli(tmp_path, "-u", "1", "--hold", "0.3s")[0] == 0
    err = capsys.readouterr().err
    assert "projection from preflight" in err
    assert (
        "spend unknown: no price for model 'mock-model' (add it under [tool.instar.prices])" in err
    )
