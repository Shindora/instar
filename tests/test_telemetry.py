from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OtlpSpan

from instar.analyze import analyze
from instar.checks import Evidence, checks, enabled, evaluate
from instar.cli import main
from instar.config import ConfigError
from instar.events import (
    CallRole,
    CheckResult,
    CheckStatus,
    Execution,
    ExecutionStatus,
    LLMCall,
    Span,
    SpanKind,
)
from instar.proxy.pricing import Price
from instar.runner import Runner, RunOptions
from instar.server import serve_in_thread
from instar.store import StoreReader
from instar.telemetry import receiver as receiver_module
from instar.telemetry import semconv
from instar.telemetry.export import ExportingSink
from instar.telemetry.normalise import TelemetryMap
from instar.telemetry.receiver import Receiver
from instar.telemetry.root import judge_span_id, split_phases

if TYPE_CHECKING:
    from collections.abc import Sequence

    from instar.events import Event

TRACED = Path(__file__).parents[1] / "tests" / "fixtures" / "agentfile_traced.py"
SESSION = "ab" * 16 + "-" + "cd" * 8


class ListSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)

    @property
    def spans(self) -> list[Span]:
        return [e for e in self.events if isinstance(e, Span)]


def kv(key: str, value: str | int) -> KeyValue:
    any_value = (
        AnyValue(int_value=value) if isinstance(value, int) else AnyValue(string_value=value)
    )
    return KeyValue(key=key, value=any_value)


def export(*spans: OtlpSpan, resource: dict[str, str] | None = None) -> bytes:
    attributes = [kv(k, v) for k, v in (resource or {}).items()]
    request = ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=Resource(attributes=attributes),
                scope_spans=[ScopeSpans(spans=list(spans))],
            )
        ]
    )
    return request.SerializeToString()


def otlp_span(trace_hex: str, name: str, **attributes: str) -> OtlpSpan:
    return OtlpSpan(
        trace_id=bytes.fromhex(trace_hex),
        span_id=bytes.fromhex("11" * 8),
        name=name,
        start_time_unix_nano=1,
        end_time_unix_nano=2,
        attributes=[kv(k, v) for k, v in attributes.items()],
    )


PROTOBUF = {"content-type": "application/x-protobuf"}


def test_receiver_joins_spans_drops_content_and_hashes_tool_arguments() -> None:
    sink = ListSink()
    known = "ef" * 16
    receiver = Receiver(sink=sink, mapping=TelemetryMap(), traces={known: SESSION})
    tool = otlp_span(
        known,
        "execute_tool search",
        **{
            semconv.OPERATION: "execute_tool",
            semconv.TOOL_NAME: "search",
            semconv.TOOL_ARGUMENTS: '{"q": "secret"}',
        },
    )
    chat = otlp_span(
        "00" * 16, "chat m", **{semconv.OPERATION: "chat", "gen_ai.input.messages": "hi"}
    )
    stray = otlp_span("99" * 16, "work")
    with serve_in_thread(receiver.build_app()) as url:
        assert (
            httpx.post(f"{url}/v1/traces", content=export(tool), headers=PROTOBUF).status_code
            == 200
        )
        body = gzip.compress(export(chat, resource={semconv.SESSION: SESSION}))
        gzipped = {**PROTOBUF, "content-encoding": "gzip"}
        assert httpx.post(f"{url}/v1/traces", content=body, headers=gzipped).status_code == 200
        assert (
            httpx.post(f"{url}/v1/traces", content=export(stray), headers=PROTOBUF).status_code
            == 200
        )
    joined_tool, joined_chat, unjoined = sink.spans
    assert (joined_tool.session, joined_tool.kind) == (SESSION, SpanKind.TOOL)
    assert semconv.TOOL_ARGUMENTS not in joined_tool.attributes
    assert len(str(joined_tool.attributes[semconv.TOOL_ARGUMENTS_HASH])) == 16
    assert (joined_chat.session, joined_chat.kind) == (SESSION, SpanKind.LLM)
    assert "gen_ai.input.messages" not in joined_chat.attributes
    assert (unjoined.session, receiver.unjoined, receiver.received) == ("", 1, 3)


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/v1/metrics", PROTOBUF, 200),
        ("/v1/logs", PROTOBUF, 200),
        ("/v1/traces", {"content-type": "application/json"}, 415),
        ("/v1/traces", {**PROTOBUF, "content-encoding": "br"}, 415),
        ("/v1/traces", PROTOBUF, 413),
    ],
)
def test_receiver_answers_every_signal_and_refuses_what_it_cannot_read(
    path: str, headers: dict[str, str], status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_module, "MAX_BODY_BYTES", 8)
    body = b"x" * (9 if status == 413 else 0)
    receiver = Receiver(sink=ListSink(), mapping=TelemetryMap(), traces={})
    with serve_in_thread(receiver.build_app()) as url:
        assert httpx.post(f"{url}{path}", content=body, headers=headers).status_code == status


def test_mapping_file_overrides_names_and_reads_cost_in_its_unit(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.toml"
    path.write_text(
        '[kinds]\noperation = "op"\nllm = ["generate"]\n'
        '[names]\ntool = "tool.id"\n[cost]\nattribute = "llm.cost"\nunit = "usd_micros"\n'
    )
    mapping = TelemetryMap.load(path)
    assert mapping.kind({"op": "generate"}) is SpanKind.LLM
    assert mapping.kind({"tool.id": "x"}) is SpanKind.TOOL
    assert mapping.kind({semconv.OPERATION: "chat"}) is SpanKind.OTHER
    assert mapping.cost_usd({"llm.cost": 2500}) == pytest.approx(0.0025)
    assert TelemetryMap().cost_usd({"llm.cost": 1}) is None
    assert TelemetryMap.from_tables(mapping.to_tables()) == mapping
    path.write_text("[colours]\nred = 1\n")
    with pytest.raises(ConfigError, match=r"unknown section \[colours\]"):
        TelemetryMap.load(path)


def span(span_id: str, parent: str | None, kind: SpanKind = SpanKind.LLM, **attrs: object) -> Span:
    return Span(
        session=SESSION,
        trace_id="ab" * 16,
        span_id=span_id,
        parent_id=parent,
        kind=kind,
        name=span_id,
        started_ns=int(span_id, 16),
        ended_ns=int(span_id, 16) + 1,
        attributes=dict(attrs),
    )


def test_judge_spans_are_told_apart_by_their_root() -> None:
    run_root, judge_root = SESSION.partition("-")[2], judge_span_id(SESSION)
    spans = [
        span("01" * 8, run_root),
        span("02" * 8, "01" * 8),
        span("03" * 8, judge_root),
        span("04" * 8, "03" * 8),
    ]
    agent, judge = split_phases(spans, SESSION)
    assert [s.span_id for s in agent] == ["01" * 8, "02" * 8]
    assert [s.span_id for s in judge] == ["03" * 8, "04" * 8]


def evidence(
    *,
    status: ExecutionStatus = ExecutionStatus.OK,
    tools: Sequence[tuple[str, str]] = (),
    calls: Sequence[tuple[int, int, int, float | None]] = (),
) -> Evidence:
    execution = Execution(
        session=SESSION,
        task_id="t",
        arm="default",
        repeat=0,
        stage=1,
        status=status,
        started_ns=0,
        ended_ns=1,
        run_ns=1,
    )
    tool_spans = [
        span(
            f"{i + 1:016x}",
            None,
            SpanKind.TOOL,
            **{semconv.TOOL_NAME: name, semconv.TOOL_ARGUMENTS_HASH: args},
        )
        for i, (name, args) in enumerate(tools)
    ]
    model_calls = [
        LLMCall(
            session=SESSION,
            role=CallRole.AGENT,
            upstream="openai",
            index=index,
            method="POST",
            path="chat/completions",
            model="m",
            status=code,
            started_ns=start,
            ended_ns=start + 1,
            cost_usd=cost,
        )
        for index, (code, start, _, cost) in enumerate(calls, start=1)
    ]
    return Evidence(execution=execution, calls=model_calls, spans=tool_spans)


def verdict(name: str, setting: object, ev: Evidence) -> tuple[CheckStatus, str | None]:
    [result] = evaluate(ev, {name: setting})
    return result.status, result.detail


def test_built_in_checks_pass_fail_and_say_when_they_cannot_judge() -> None:
    passed, failed, unknown = CheckStatus.PASSED, CheckStatus.FAILED, CheckStatus.NOT_EVALUATED
    assert verdict("terminated", True, evidence())[0] is passed
    assert verdict("terminated", True, evidence(status=ExecutionStatus.TIMEOUT))[0] is failed
    calls = [(200, 0, 0, 0.01)] * 3
    assert verdict("max_llm_calls", 2, evidence(calls=calls))[0] is failed
    assert verdict("cost_cap", 0.05, evidence(calls=calls))[0] is passed
    assert verdict("cost_cap", 0.05, evidence(calls=[(200, 0, 0, None)]))[0] is unknown
    quick = [(429, 0, 0, 0.0), (200, 100_000_000, 0, 0.0)]
    patient = [(429, 0, 0, 0.0), (200, 2_000_000_000, 0, 0.0)]
    assert verdict("backoff_after_429", 1.0, evidence(calls=quick))[0] is failed
    assert verdict("backoff_after_429", 1.0, evidence(calls=patient))[0] is passed
    assert verdict("backoff_after_429", 1.0, evidence(calls=calls))[0] is unknown
    same = [("search", "aa")] * 4
    varied = [("search", "aa"), ("search", "bb")] * 2
    assert verdict("no_tool_loop", 4, evidence(tools=same))[0] is failed
    assert verdict("no_tool_loop", 4, evidence(tools=varied))[0] is passed
    assert verdict("forbidden_tools", ["rm"], evidence(tools=[("rm", "x")])) == (
        failed,
        "called rm",
    )
    assert verdict("max_steps", 3, evidence(tools=same))[0] is failed
    assert verdict("max_steps", 3, evidence())[0] is unknown
    assert enabled({"max_steps": False, "forbidden_tools": ["rm"]})["forbidden_tools"] == ["rm"]
    assert "max_steps" not in enabled({"max_steps": False})
    assert "cost_cap" in checks


def traced_run(tmp_path: Path, user: str, **options: object) -> tuple[Path, StoreReader]:
    mapping = tmp_path / "telemetry.toml"
    mapping.write_text('[cost]\nattribute = "test.cost_usd"\nunit = "usd"\n')
    run = RunOptions(
        agentfile=TRACED,
        project="traced",
        runs_dir=tmp_path,
        user=user,
        users=4,
        checks={"max_steps": 10, "forbidden_tools": ["delete_file"]},
        telemetry_map=mapping,
        prices={"mock-model": Price(input=1.0, output=2.0)},
        span_grace_s=0.3,
        **options,  # ty: ignore[invalid-argument-type]  overrides by field name
    )
    result = asyncio.run(Runner(run).run())
    return result.run_dir, StoreReader(result.run_dir)


def outcomes(reader: StoreReader) -> dict[tuple[str, str], CheckStatus]:
    task_of = {e.session: e.task_id for e in reader.select(Execution)}
    return {(task_of[r.session], r.check): r.status for r in reader.select(CheckResult)}


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
def test_checks_pass_and_fail_as_scripted_with_spans(tmp_path: Path) -> None:
    run_dir, reader = traced_run(tmp_path, "TracedBot")
    found = outcomes(reader)
    failed, passed = CheckStatus.FAILED, CheckStatus.PASSED
    assert {t: found[t, "no_tool_loop"] for t in ("calm", "loop")} == {
        "calm": passed,
        "loop": failed,
    }
    assert found["forbidden", "forbidden_tools"] is failed
    assert found["long", "max_steps"] is failed
    assert found["calm", "max_steps"] is passed
    assert all(found[t, "terminated"] is passed for t in ("calm", "loop", "forbidden", "long"))
    calm = next(e.session for e in reader.select(Execution) if e.task_id == "calm")
    [steps] = [r for r in reader.select(CheckResult, session=calm) if r.check == "max_steps"]
    assert steps.detail == "2 steps"
    assert all(s.session for s in reader.select(Span))
    assert not any("gen_ai.input.messages" in s.attributes for s in reader.select(Span))

    analysis = analyze(run_dir, RunOptions())
    assert analysis.agent_cost_usd == pytest.approx(13 * 5e-6)
    assert analysis.summary.cost_usd == pytest.approx(13 * 5e-6)
    assert analysis.cost_mismatch is None
    summary = {c.name: c for c in analysis.checks}
    assert summary["no_tool_loop"].failing_tasks == ["loop"]


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
def test_without_spans_span_checks_are_not_evaluated_and_the_run_succeeds(tmp_path: Path) -> None:
    run_dir, reader = traced_run(tmp_path, "QuietBot")
    found = outcomes(reader)
    assert {found[t, "max_steps"] for t in ("calm", "loop")} == {CheckStatus.NOT_EVALUATED}
    assert found["calm", "terminated"] is CheckStatus.PASSED
    assert all(e.status is ExecutionStatus.OK for e in reader.select(Execution))
    assert analyze(run_dir, RunOptions()).agent_cost_usd is None


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
def test_a_cost_gap_between_spans_and_prices_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRACE_COST", "1e-5")
    run_dir, _ = traced_run(tmp_path, "TracedBot", limit=1)
    mismatch = analyze(run_dir, RunOptions()).cost_mismatch
    assert mismatch is not None
    assert "differs from the priced cost $0.000005 by 50%" in mismatch


@pytest.mark.usefixtures("model_env")
@pytest.mark.xdist_group("processes")
@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("TracedBot", "spans for this execution: agent 1, llm 1, tool 1; judge spans kept out"),
        ("QuietBot", "no spans arrived for this execution: span checks"),
    ],
)
def test_doctor_reports_spans_and_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], user: str, expected: str
) -> None:
    assert main(["doctor", "-f", str(TRACED), "--user", user, "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert expected in out
    assert "spans not joined to any execution: 0" in out
    assert "check terminated: passed" in out


def test_proxy_spans_are_forwarded_to_an_external_collector() -> None:
    collected = ListSink()
    collector = Receiver(sink=collected, mapping=TelemetryMap(), traces={})
    with serve_in_thread(collector.build_app()) as url:
        stored = ListSink()
        sink = ExportingSink(stored, url)
        call = LLMCall(
            session=SESSION,
            role=CallRole.AGENT,
            upstream="openai",
            index=1,
            method="POST",
            path="chat/completions",
            model="m",
            status=200,
            started_ns=1_000,
            ended_ns=2_000,
            input_tokens=7,
            output_tokens=3,
        )
        sink.emit(call)
        sink.close()
    assert stored.events == [call]
    [forwarded] = collected.spans
    assert (forwarded.session, forwarded.kind, forwarded.name) == (SESSION, SpanKind.LLM, "chat m")
    assert forwarded.trace_id == SESSION.partition("-")[0]
    assert forwarded.parent_id == SESSION.partition("-")[2]
    assert forwarded.attributes[semconv.OUTPUT_TOKENS] == 3


def test_the_check_api_is_public() -> None:
    import instar

    assert instar.check is not None
    assert json.dumps(sorted(checks.names()))
