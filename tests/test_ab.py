from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from instar.analyze.ab import compare_arms, mcnemar_exact
from instar.analyze.data import RunData
from instar.cli import main
from instar.events import Execution, ExecutionStatus, LLMCall
from instar.proxy.faults import pick_fault
from instar.runner import Arm, ordered_arms
from instar.store import StoreReader

if TYPE_CHECKING:
    from collections.abc import Sequence

AB = Path(__file__).parents[1] / "tests" / "fixtures" / "agentfile_ab.py"


def ab_run(tmp_path: Path, user: str, *args: str) -> tuple[int, Path]:
    runs = tmp_path / "runs"
    argv = ["run", "-f", str(AB), "--user", user, "-u", "8", "--runs-dir", str(runs), *args]
    code = main([*argv, "--project", "ab"])
    [run_dir] = (runs / "ab").iterdir()
    return code, run_dir


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
def test_a_better_arm_is_detected_with_its_difference(tmp_path: Path) -> None:
    code, run_dir = ab_run(tmp_path, "VersionBot", "--ab", "AGENT_VERSION=v1,v2")
    assert code == 0
    report = json.loads((run_dir / "report.json").read_text())
    ab = report["ab"]
    assert ab["a"]["primary"]["value"] == 0.5
    assert ab["b"]["primary"]["value"] == 1.0
    diff = ab["primary"]
    assert diff["value"] == pytest.approx(0.5)
    assert diff["detected"] is True
    assert diff["low"] > 0
    assert (diff["pairs"], diff["tasks"]) == (20, 20)
    assert diff["mcnemar_p"] == pytest.approx(mcnemar_exact(10, 0))
    assert "AGENT_VERSION=v2 does better than v1: passed +50%" in ab["sentence"]

    arms = json.loads((run_dir / "config.json").read_text())["arms"]
    first, second = arms["v1"], arms["v2"]
    assert {k for k in first.keys() | second.keys() if first.get(k) != second.get(k)} == {
        "AGENT_VERSION"
    }
    executions = StoreReader(run_dir).select(Execution)
    pairs = defaultdict(set)
    for execution in executions:
        pairs[execution.task_id, execution.repeat].add(execution.arm)
    assert len(pairs) == 20
    assert all(arms == {"v1", "v2"} for arms in pairs.values())


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
def test_identical_arms_report_no_difference_and_what_could_be_detected(tmp_path: Path) -> None:
    code, run_dir = ab_run(tmp_path, "VersionBot", "--ab", "AGENT_VERSION=v1,v1-copy")
    assert code == 0
    ab = json.loads((run_dir / "report.json").read_text())["ab"]
    assert ab["a"]["primary"]["value"] == ab["b"]["primary"]["value"] == 0.5
    assert ab["primary"]["detected"] is False
    assert ab["primary"]["smallest_detectable"] == pytest.approx(6 / 20)
    assert (
        "No difference detected between v1 and v1-copy: passed +0% (+0% to +0%)" in ab["sentence"]
    )
    assert "could detect a difference of about 30% or more" in ab["sentence"]
    assert "No difference detected" in (run_dir / "report.md").read_text()


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
def test_both_arms_meet_the_same_fault_decision_on_each_call(tmp_path: Path) -> None:
    args = ["--ab", "AGENT_VERSION=x,y", "--faults", "http_500=0.3,http_429=0.2", "--seed", "11"]
    code, run_dir = ab_run(tmp_path, "StubbornBot", *args)
    assert code == 0
    reader = StoreReader(run_dir)
    sessions = {(e.task_id, e.repeat, e.arm): e.session for e in reader.select(Execution)}
    calls: dict[str, list[LLMCall]] = defaultdict(list)
    for call in reader.select(LLMCall):
        calls[call.session].append(call)
    fired = 0
    for task in (f"t{i:02d}" for i in range(10)):
        x = sorted(calls[sessions[task, 0, "x"]], key=lambda c: c.index)
        y = sorted(calls[sessions[task, 0, "y"]], key=lambda c: c.index)
        for call_x, call_y in zip(x, y, strict=False):
            assert call_x.index == call_y.index
            assert call_x.fault == call_y.fault
            assert call_x.fault == pick_fault(
                {"http_500": 0.3, "http_429": 0.2}, seed=11, key=f"{task}:0", index=call_x.index
            )
            fired += call_x.fault is not None
    assert fired > 0


@pytest.mark.parametrize(
    "args",
    [
        ["--ab", "AGENT_VERSION=v1,v1"],
        ["--ab", "AGENT_VERSION"],
        ["--ab", "AGENT_VERSION=v1,v2", "--executor", "inline"],
    ],
)
def test_bad_ab_options_exit_with_code_2(tmp_path: Path, args: list[str]) -> None:
    runs = tmp_path / "runs"
    assert main(["run", "-f", str(AB), "--user", "VersionBot", "--runs-dir", str(runs), *args]) == 2


def test_arm_order_is_seeded_per_task_and_varies() -> None:
    arms = [Arm(name="a", env={}), Arm(name="b", env={})]
    firsts = [ordered_arms(arms, seed=3, task_id=f"t{i}", repeat=0)[0].name for i in range(40)]
    again = [ordered_arms(arms, seed=3, task_id=f"t{i}", repeat=0)[0].name for i in range(40)]
    assert firsts == again
    assert set(firsts) == {"a", "b"}


def synthetic(rows: Sequence[tuple[str, int, str, bool | None, ExecutionStatus]]) -> RunData:
    """Executions from (task, repeat, arm, passed, status); passed None means no verdict."""
    executions, metrics = [], {}
    for n, (task, repeat, arm, passed, status) in enumerate(rows):
        session = f"s{n}"
        executions.append(
            Execution(
                session=session,
                task_id=task,
                arm=arm,
                repeat=repeat,
                stage=1,
                status=status,
                started_ns=n,
                ended_ns=n + 1,
                run_ns=1,
            )
        )
        if passed is not None:
            metrics[session] = {"passed": passed}
    return RunData(
        run=None, executions=executions, metrics=metrics, calls=[], stages=[], loads=[], sheds=[]
    )


def test_errors_and_timeouts_count_as_failures_and_are_never_dropped() -> None:
    ok, error, timeout = ExecutionStatus.OK, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT
    rows = []
    for i in range(4):
        rows.append((f"t{i}", 0, "a", True, ok))
        rows.append((f"t{i}", 0, "b", None, error if i % 2 else timeout))
    result = compare_arms(synthetic(rows), variable="V", a="a", b="b", primary="passed", seed="x")
    assert result.b.executions == 4
    assert result.b.primary.value == 0.0
    assert result.primary is not None
    assert result.primary.value == -1.0
    assert result.primary.pairs == 4


def test_bootstrap_resamples_tasks_and_mcnemar_needs_one_pair_per_task() -> None:
    ok = ExecutionStatus.OK
    rows = []
    for repeat in range(10):
        rows += [("better", repeat, "a", False, ok), ("better", repeat, "b", True, ok)]
        rows += [("same", repeat, "a", True, ok), ("same", repeat, "b", True, ok)]
    result = compare_arms(synthetic(rows), variable="V", a="a", b="b", primary="passed", seed="x")
    diff = result.primary
    assert diff is not None
    assert diff.value == 0.5
    assert (diff.low, diff.high) == (0.0, 1.0)
    assert diff.detected is False
    assert diff.mcnemar_p is None
    assert (diff.pairs, diff.tasks) == (20, 2)


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
