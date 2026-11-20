from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from instar.cli import main
from instar.events import Execution, ExecutionStatus, LLMCall, Metric
from instar.runner import MAX_PROCESS_USERS, Runner, RunOptions, RunResult
from instar.store import StoreReader

ROOT = Path(__file__).parents[1]
QUICKSTART = ROOT / "examples" / "quickstart" / "agentfile.py"
BOTS = ROOT / "tests" / "fixtures" / "agentfile_bots.py"
FAULTY = ROOT / "tests" / "fixtures" / "agentfile_faulty.py"
BROKEN = ROOT / "tests" / "fixtures" / "agentfile_broken.py"

pytestmark = pytest.mark.usefixtures("model_env")


def run(agentfile: Path, runs_dir: Path, **options: object) -> tuple[RunResult, StoreReader]:
    runner = Runner(RunOptions(agentfile=agentfile, project="test", runs_dir=runs_dir, **options))  # ty: ignore[invalid-argument-type]  options are passed through
    result = asyncio.run(runner.run())
    return result, StoreReader(result.run_dir)


def only_run_dir(runs_dir: Path, project: str) -> Path:
    [run_dir] = (runs_dir / project).iterdir()
    return run_dir


def assert_attributed(reader: StoreReader, execution: Execution, calls: int) -> None:
    own = reader.select(LLMCall, session=execution.session)
    assert len(own) == calls
    assert all(c.status == 200 for c in own)


def test_quickstart_run_joins_every_execution_to_its_calls_and_metrics(tmp_path: Path) -> None:
    argv = ["run", "-f", str(QUICKSTART), "-u", "5", "--repeat", "2"]
    assert main([*argv, "--runs-dir", str(tmp_path), "--project", "qs"]) == 0
    run_dir = only_run_dir(tmp_path, "qs")
    reader = StoreReader(run_dir)
    executions = reader.select(Execution)
    pairs = {(e.task_id, e.repeat) for e in executions}
    assert pairs == {(t, r) for t in ("capital", "colour", "number") for r in (0, 1)}
    assert len(executions) == 6
    for execution in executions:
        assert execution.status is ExecutionStatus.OK
        assert_attributed(reader, execution, calls=1)
        assert reader.select(Metric, session=execution.session) == [
            Metric(session=execution.session, name="passed", value=True)
        ]
        assert (run_dir / "logs" / f"{execution.session}.log").exists()
    assert reader.select(LLMCall, session="") == []
    assert (run_dir / "config.json").exists()


def test_import_time_reader_is_attributed_in_process_mode(tmp_path: Path) -> None:
    result, reader = run(FAULTY, tmp_path, user="ImportTimeBot", users=6)
    executions = reader.select(Execution)
    assert result.statuses == {ExecutionStatus.OK: 6}
    for execution in executions:
        assert_attributed(reader, execution, calls=1)
    assert len({e.session for e in executions}) == 6


def test_raise_hang_and_exit_are_recorded_and_the_run_continues(tmp_path: Path) -> None:
    result, reader = run(FAULTY, tmp_path, user="FaultyBot", users=4, timeout_s=4.0)
    by_task = {e.task_id: e for e in reader.select(Execution)}
    assert {task: e.status for task, e in by_task.items()} == {
        "ok": ExecutionStatus.OK,
        "raise": ExecutionStatus.ERROR,
        "hang": ExecutionStatus.TIMEOUT,
        "exit": ExecutionStatus.CRASHED,
    }
    assert "agent failed on purpose" in (by_task["raise"].error or "")
    assert "code 17" in (by_task["exit"].error or "")
    assert by_task["hang"].error is None
    assert result.cancelled is False


@pytest.mark.parametrize("executor", ["process", "inline"])
@pytest.mark.parametrize("user", ["SyncBot", "AsyncBot"])
def test_sync_and_async_run_and_judge_in_both_executors(
    tmp_path: Path, executor: str, user: str
) -> None:
    result, reader = run(BOTS, tmp_path, user=user, executor=executor, users=3)
    assert result.statuses == {ExecutionStatus.OK: 3}
    for execution in reader.select(Execution):
        assert_attributed(reader, execution, calls=1)
        metrics = {m.name: m.value for m in reader.select(Metric, session=execution.session)}
        assert metrics["passed"] is True
        if user == "AsyncBot":
            assert metrics["length"] == len("echo: hello 0")


@pytest.mark.parametrize("keep", [True, False])
def test_outputs_are_stored_only_when_asked(tmp_path: Path, keep: bool) -> None:
    _, reader = run(BOTS, tmp_path, user="SyncBot", executor="inline", keep_outputs=keep, limit=1)
    [execution] = reader.select(Execution)
    assert execution.output == ("echo: hello 0" if keep else None)


async def test_stop_cancels_in_flight_executions_and_closes_the_run(tmp_path: Path) -> None:
    runner = Runner(
        RunOptions(agentfile=FAULTY, project="test", runs_dir=tmp_path, user="HangBot", users=3)
    )
    started = time.monotonic()
    running = asyncio.create_task(runner.run())
    while sum(1 for _ in (runner.run_dir / "logs").glob("*.log")) < 3:
        assert time.monotonic() - started < 15, "executions never started"
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)
    runner.stop()
    result = await asyncio.wait_for(running, timeout=10)
    assert result.cancelled is True
    assert result.statuses == {ExecutionStatus.CANCELLED: 3}
    reader = StoreReader(result.run_dir)
    assert [e.status for e in reader.select(Execution)] == [ExecutionStatus.CANCELLED] * 3


def test_debug_prints_calls_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["run", "--debug", "-f", str(BOTS), "--user", "SyncBot", "--runs-dir", str(tmp_path)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "openai POST chat/completions -> 200" in out
    assert 'output: "echo: hello 0"' in out
    assert "1 executions" in out


def doctor(capsys: pytest.CaptureFixture[str], tmp_path: Path, *args: str) -> tuple[int, str]:
    code = main(["doctor", "--runs-dir", str(tmp_path), *args])
    return code, capsys.readouterr().out


def test_doctor_passes_the_quickstart(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = doctor(capsys, tmp_path, "-f", str(QUICKSTART))
    assert code == 0, out
    assert "model calls attributed to this session: 1 (openai x1)" in out
    assert "judge: passed=True" in out
    assert "leaves room" in out


def test_doctor_flags_an_inline_run_whose_calls_carry_no_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = doctor(
        capsys, tmp_path, "-f", str(BOTS), "--user", "EnvBot", "--executor", "inline"
    )
    assert code == 3
    assert "no calls arrived for this session" in out
    assert "1 model calls arrived without a session" in out


def test_doctor_warns_when_timeout_is_below_three_run_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = doctor(
        capsys, tmp_path, "-f", str(FAULTY), "--user", "SlowBot", "--timeout-s", "1.5"
    )
    assert code == 0, out
    assert "WARNING: timeout_s = 1.5 s is below 3x the measured run time" in out


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["-f", "missing.py"], 2, "agentfile not found"),
        (["-f", str(BROKEN)], 3, "broken on purpose"),
        (["-f", str(BOTS)], 2, "several AgentUser classes (AsyncBot, EnvBot, SyncBot)"),
        (["-f", str(BOTS), "--user", "Nope"], 2, "no AgentUser class 'Nope'"),
    ],
)
def test_agentfile_problems_exit_with_clear_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str], code: int, message: str
) -> None:
    assert main(["run", "--runs-dir", str(tmp_path), *args]) == code
    assert message in capsys.readouterr().err


def test_process_users_are_capped(tmp_path: Path) -> None:
    runner = Runner(RunOptions(agentfile=BOTS, project="test", runs_dir=tmp_path, users=100))
    assert runner.users == MAX_PROCESS_USERS
    assert "users capped at 64" in runner.warnings[0]
