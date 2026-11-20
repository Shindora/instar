from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn

from instar.doctor import diagnose
from instar.events import Execution, LLMCall, Metric
from instar.executors import executors
from instar.proxy.app import Proxy
from instar.proxy.faults import check_rates
from instar.proxy.upstreams import resolve_upstreams
from instar.runner import DEFAULT_TIMEOUT_S, DEFAULT_USERS, Runner, RunOptions, RunResult
from instar.server import KEEP_ALIVE_S
from instar.store import JsonlSink, StoreReader
from instar.user import AgentfileError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _pair(value: str) -> tuple[str, str]:
    name, sep, rest = value.partition("=")
    if not sep or not name or not rest:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    return name.strip(), rest.strip()


def _rates(value: str) -> dict[str, float]:
    rates: dict[str, float] = {}
    for item in filter(None, value.split(",")):
        name, raw = _pair(item)
        try:
            rates[name] = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(f"fault rate must be a number: {item!r}") from None
    try:
        check_rates(rates)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return rates


def _positive[T: (int, float)](kind: type[T]) -> Callable[[str], T]:
    def parse(value: str) -> T:
        try:
            number = kind(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
        if number <= 0:
            raise argparse.ArgumentTypeError(f"must be above 0, got {value}")
        return number

    return parse


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-f", "--agentfile", type=Path, default=Path("agentfile.py"))
    parser.add_argument("--user", help="AgentUser class to run, when the agentfile has several")
    parser.add_argument("--project", default=Path.cwd().name, help="names runs/<project>/")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--executor", choices=executors.names(), default="process")
    parser.add_argument("--timeout-s", type=_positive(float), default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instar", description="Pre-release testing for AI agents."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the agentfile's tasks and record everything")
    _add_run_options(run)
    run.add_argument("-u", "--users", type=_positive(int), default=DEFAULT_USERS)
    run.add_argument("--repeat", type=_positive(int), default=1)
    run.add_argument("--limit", type=_positive(int), help="run only the first N tasks")
    run.add_argument("--faults", type=_rates, default={}, metavar="NAME=RATE[,...]")
    run.add_argument("--latency-s", type=float, default=5.0, help="delay of the latency fault")
    run.add_argument("--spend-cap", type=_positive(float), help="USD; counts reported cost only")
    run.add_argument("--keep-outputs", action="store_true", help="store each agent output")
    run.add_argument("--debug", action="store_true", help="one task, inline; print calls, output")
    run.set_defaults(handler=_run)

    doctor = commands.add_parser("doctor", help="preflight: run one task, report what arrived")
    _add_run_options(doctor)
    doctor.set_defaults(handler=_doctor)

    proxy = commands.add_parser(
        "proxy",
        help="run the LLM proxy on its own, for agents that are not Python functions",
        description="Point an agent's base URL at http://HOST:PORT/s/<session>/<upstream>.",
    )
    proxy.add_argument("--upstream", type=_pair, action="append", default=[], metavar="NAME=URL")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8080)
    proxy.add_argument("--faults", type=_rates, default={}, metavar="NAME=RATE[,...]")
    proxy.add_argument("--latency-s", type=float, default=5.0, help="delay of the latency fault")
    proxy.add_argument("--seed", type=int, default=0)
    proxy.add_argument("--out", type=Path, default=Path("llm_calls.jsonl"), help="JSONL event log")
    proxy.set_defaults(handler=_run_proxy)
    return parser


def _options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        agentfile=args.agentfile,
        project=args.project,
        user=args.user,
        runs_dir=args.runs_dir,
        executor=args.executor,
        timeout_s=args.timeout_s,
        seed=args.seed,
    )


def _start(runner: Runner) -> RunResult | int:
    """Run to the end; return the result, or the exit code when the run could not finish."""
    for warning in runner.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    try:
        return asyncio.run(runner.run())
    except AgentfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print(f"run cancelled; partial results in {runner.run_dir}", file=sys.stderr)
        return 3


def _summary(result: RunResult) -> str:
    total = sum(result.statuses.values())
    parts = [f"{status} {n}" for status, n in sorted(result.statuses.items())]
    reader = StoreReader(result.run_dir)
    primary = [m.value for m in reader.select(Metric, name=result.primary)]
    reader.close()
    if primary and all(isinstance(v, bool) for v in primary):
        passed = sum(1 for v in primary if v)
        parts.append(f"{result.primary} {passed}/{len(primary)} ({passed / len(primary):.0%})")
    return f"run {result.run_id}: {total} executions · " + " · ".join(parts)


def _print_debug(result: RunResult) -> None:
    reader = StoreReader(result.run_dir)
    calls = reader.select(LLMCall)
    executions = reader.select(Execution)
    reader.close()
    print("calls:")
    for c in calls:
        where = "" if c.session else "  (no session)"
        timing = f"{(c.ended_ns - c.started_ns) / 1e9:.2f} s"
        tokens = f"{c.input_tokens} in / {c.output_tokens} out tokens"
        fault = f"  [fault: {c.fault}]" if c.fault else ""
        print(f"  {c.upstream} {c.method} {c.path} -> {c.status}  {timing}  {tokens}{fault}{where}")
    if not calls:
        print("  none arrived at the proxy")
    for execution in executions:
        print(f"output: {json.dumps(execution.output, default=str)}")
        if execution.error:
            print(f"error:\n{execution.error}")


def _run(args: argparse.Namespace) -> int:
    options = copy.replace(
        _options(args),
        users=args.users,
        repeat=args.repeat,
        limit=args.limit,
        faults=args.faults,
        latency_s=args.latency_s,
        spend_cap_usd=args.spend_cap,
        keep_outputs=args.keep_outputs,
    )
    if args.debug:
        options = copy.replace(
            options, executor="inline", users=1, repeat=1, limit=1, keep_outputs=True
        )
    result = _start(Runner(options))
    if isinstance(result, int):
        return result
    if args.debug:
        _print_debug(result)
    print(_summary(result))
    print(f"results in {result.run_dir}")
    if result.cancelled:
        print("run cancelled before every task ran", file=sys.stderr)
        return 3
    return 0


def _doctor(args: argparse.Namespace) -> int:
    options = copy.replace(_options(args), users=1, repeat=1, limit=1)
    result = _start(Runner(options))
    if isinstance(result, int):
        return result
    diagnosis = diagnose(options, result)
    print("\n".join(diagnosis.lines))
    print("doctor: ok" if not diagnosis.problems else f"doctor: {diagnosis.problems} problem(s)")
    return diagnosis.exit_code


def _run_proxy(args: argparse.Namespace) -> int:
    upstreams = resolve_upstreams(dict(args.upstream))
    with closing(JsonlSink(args.out)) as sink:
        proxy = Proxy(
            upstreams=upstreams,
            sink=sink,
            seed=args.seed,
            fault_rates=args.faults,
            latency_s=args.latency_s,
        )
        routes = ", ".join(f"{name}={url}" for name, url in sorted(upstreams.items()))
        print(f"instar proxy on http://{args.host}:{args.port} ({routes})", file=sys.stderr)
        uvicorn.run(
            proxy.build_app(),
            host=args.host,
            port=args.port,
            timeout_keep_alive=KEEP_ALIVE_S,
            log_level="warning",
            access_log=False,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
