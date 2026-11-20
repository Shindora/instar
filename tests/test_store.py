from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from instar.events import (
    CallRole,
    CheckResult,
    CheckStatus,
    Execution,
    ExecutionStatus,
    LLMCall,
    Metric,
    Run,
    Span,
    SpanKind,
)
from instar.store import JSONL_NAME, Store, StoreReader

if TYPE_CHECKING:
    from pathlib import Path

    from instar.events import Event


def session_events(session: str) -> list[Event]:
    calls: list[Event] = [
        LLMCall(
            session=session,
            role=CallRole.AGENT,
            upstream="openai",
            index=i,
            method="POST",
            path="chat/completions",
            model="m",
            status=200,
            started_ns=1_700_000_000_000_000_000 + i,
            ended_ns=1_700_000_000_000_000_100 + i,
            usage={"prompt_tokens": 7, "prompt_tokens_details.cached_tokens": 5},
            input_tokens=7,
            cached_input_tokens=5,
            output_tokens=3,
        )
        for i in range(1, 21)
    ]
    return [
        *calls,
        Execution(
            session=session,
            task_id=f"task-{session}",
            arm="default",
            repeat=0,
            stage=1,
            status=ExecutionStatus.OK,
            started_ns=1,
            run_ns=2,
            ramping=True,
        ),
        Metric(session=session, name="passed", value=True),
        Metric(session=session, name="score", value=0.75),
        Metric(session=session, name="label", value="good"),
        Span(
            session=session,
            trace_id="0" * 32,
            span_id="1" * 16,
            parent_id=None,
            kind=SpanKind.TOOL,
            name="execute_tool lookup",
            started_ns=1,
            ended_ns=2,
            attributes={"gen_ai.tool.name": "lookup", "retries": 2},
        ),
        CheckResult(session=session, check="max_steps", status=CheckStatus.NOT_EVALUATED),
    ]


async def test_concurrent_writes_all_arrive_and_read_back_by_session(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run = Run(run_id="r1", project="demo", started_ns=1, config={"seed": 7, "users": 4})
    store.emit(run)

    async def coroutine_producer(session: str) -> None:
        for event in session_events(session):
            store.emit(event)
            await asyncio.sleep(0)

    def thread_producer(session: str) -> None:
        for event in session_events(session):
            store.emit(event)

    threads = [threading.Thread(target=thread_producer, args=(f"t{i}",)) for i in range(10)]
    for thread in threads:
        thread.start()
    await asyncio.gather(*(coroutine_producer(f"c{i}") for i in range(50)))
    for thread in threads:
        thread.join()
    finished = copy.replace(run, ended_ns=99)
    store.emit(finished)
    store.close()

    reader = StoreReader(tmp_path)
    per_session = len(session_events("x"))
    counts = {kind: len(reader.select(kind)) for kind in (LLMCall, Execution, Metric, Span)}
    assert counts == {LLMCall: 60 * 20, Execution: 60, Metric: 60 * 3, Span: 60}
    assert len(reader.select(CheckResult)) == 60
    assert reader.select(Run) == [finished]
    assert len((tmp_path / JSONL_NAME).read_text().splitlines()) == 60 * per_session + 2

    for session in ("c7", "t3"):
        stored = [
            *reader.select(LLMCall, session=session),
            *reader.select(Execution, session=session),
            *reader.select(Metric, session=session),
            *reader.select(Span, session=session),
            *reader.select(CheckResult, session=session),
        ]
        assert stored == session_events(session)
    [execution] = reader.select(Execution, session="c7")
    assert execution.status is ExecutionStatus.OK
    assert execution.ramping is True
    [passed] = reader.select(Metric, session="c7", name="passed")
    assert passed.value is True
    reader.close()


def test_reader_is_read_only_and_rejects_unknown_fields(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.emit(Metric(session="s", name="passed", value=False))
    store.close()
    assert sorted(f.name for f in tmp_path.iterdir()) == [JSONL_NAME, "run.sqlite"]
    reader = StoreReader(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.db.execute("DELETE FROM metrics")
    with pytest.raises(ValueError, match="no field nope"):
        reader.select(Metric, nope=1)
    reader.close()


def test_reader_sees_events_while_the_run_is_still_writing(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.emit(Metric(session="live", name="passed", value=True))
    deadline = time.monotonic() + 5
    while not (tmp_path / "run.sqlite").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    reader = StoreReader(tmp_path)
    while not reader.select(Metric, session="live"):
        assert time.monotonic() < deadline, "event never became visible"
        time.sleep(0.01)
    store.emit(Metric(session="live", name="score", value=1.0))
    while len(reader.select(Metric, session="live")) < 2:
        assert time.monotonic() < deadline, "second event never became visible"
        time.sleep(0.01)
    reader.close()
    store.close()


@dataclass(frozen=True)
class Unregistered:
    session: str


def test_a_bad_event_is_reported_and_the_rest_are_kept(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.emit(Metric(session="s", name="passed", value=True))
    store.emit(Unregistered(session="s"))  # ty: ignore[invalid-argument-type]  deliberately wrong
    store.emit(Metric(session="s", name="score", value=0.5))
    with pytest.raises(RuntimeError, match="1 events were not stored; first: Unregistered"):
        store.close()
    reader = StoreReader(tmp_path)
    assert [m.name for m in reader.select(Metric)] == ["passed", "score"]
    reader.close()
    lines = [json.loads(line)["type"] for line in (tmp_path / JSONL_NAME).read_text().splitlines()]
    assert lines == ["Metric", "Unregistered", "Metric"]


def test_emit_after_close_fails_loudly(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.emit(Metric(session="s", name="passed", value=True))
