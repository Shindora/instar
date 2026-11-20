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

from instar.analyze import analyze
from instar.analyze import baseline as baselines
from instar.analyze.data import RunData
from instar.analyze.gate import gate_exit_code, parse_all
from instar.analyze.report import RENDERERS, write_reports
from instar.config import ConfigError, Stage, load_project, resolve
from instar.doctor import diagnose
from instar.events import Execution, LLMCall
from instar.executors import executors
from instar.proxy.app import Proxy
from instar.proxy.upstreams import resolve_upstreams
from instar.runner import Runner, RunOptions, RunResult
from instar.server import KEEP_ALIVE_S
from instar.store import JsonlSink, StoreReader
from instar.user import AgentfileError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from instar.analyze import Analysis
    from instar.analyze.baseline import Baseline

CONTROL = frozenset({"command", "handler", "profile", "debug"})
NOT_GIVEN = argparse.SUPPRESS


def _flag(parser: argparse.ArgumentParser, *names: str, key: str, **kwargs: object) -> None:
    parser.add_argument(*names, dest=key, default=NOT_GIVEN, **kwargs)  # ty: ignore[invalid-argument-type]  argparse kwargs


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="named profile from the project config")
    _flag(parser, "-f", "--agentfile", key="agentfile", help="default: agentfile.py")
    _flag(parser, "--user", key="user", help="AgentUser class, when the agentfile has several")
    _flag(parser, "--project", key="project", help="names runs/<project>/; default: this folder")
    _flag(parser, "--runs-dir", key="runs-dir", help="default: runs")
    _flag(parser, "--executor", key="executor", choices=executors.names())
    _flag(parser, "--timeout-s", key="timeout-s", help="per execution; default 600")
    _flag(parser, "--seed", key="seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instar", description="Pre-release testing for AI agents."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the agentfile's tasks and record everything")
    _add_common(run)
    _flag(run, "-u", "--users", key="users", help="concurrent users; default 4")
    _flag(run, "--stages", key="stages", help="e.g. 1,4,8: each ends when every task has run")
    _flag(run, "--hold", key="hold", help="make the stage timed, e.g. 30m")
    _flag(run, "--ramp", key="ramp", help="users added or removed between stages, e.g. 2/min")
    _flag(run, "--think-time", key="think-time", help="pause between one user's tasks, e.g. 10s")
    _flag(run, "--rate", key="rate", help="open loop: arrivals, e.g. 120/h (needs --hold)")
    _flag(run, "--max-users", key="max-users", help="open loop: arrivals over this are shed")
    _flag(run, "--repeat", key="repeat", help="runs of each task in a stage; default 1")
    _flag(run, "--limit", key="limit", help="run only the first N tasks")
    _flag(run, "--faults", key="faults", metavar="NAME=RATE[,...]")
    _flag(run, "--latency-s", key="latency-s", help="delay of the latency fault; default 5")
    _flag(run, "--spend-cap", key="spend-cap", help="USD; counts reported cost only")
    _flag(run, "--keep-outputs", key="keep-outputs", action="store_const", const=True)
    _flag(run, "--fail-if", key="fail-if", action="append", metavar="EXPR")
    _flag(run, "--baseline", key="baseline", help="baseline file for gates and per-task diffs")
    _flag(run, "--strict", key="strict", action="store_const", const=True)
    _flag(run, "--interval", key="interval", help="time series bucket; default 30s")
    _flag(run, "--breaking-p95", key="breaking-p95", help="p95 multiple that breaks; default 2")
    run.add_argument("--debug", action="store_true", help="one task, inline; print calls, output")
    run.set_defaults(handler=_run)

    doctor = commands.add_parser("doctor", help="preflight: run one task, report what arrived")
    _add_common(doctor)
    doctor.set_defaults(handler=_doctor)

    report = commands.add_parser("report", help="print a finished run's report")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--format", choices=sorted(RENDERERS), default="md")
    report.set_defaults(handler=_report)

    baseline = commands.add_parser("baseline", help="save a run as the baseline to compare with")
    baseline.add_argument("action", choices=["save"])
    baseline.add_argument("run_dir", type=Path)
    baseline.add_argument("path", type=Path)
    baseline.set_defaults(handler=_baseline)

    proxy = commands.add_parser(
        "proxy",
        help="run the LLM proxy on its own, for agents that are not Python functions",
        description="Point an agent's base URL at http://HOST:PORT/s/<session>/<upstream>.",
    )
    proxy.add_argument("--upstream", action="append", default=[], metavar="NAME=URL")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8080)
    proxy.add_argument("--faults", default="", metavar="NAME=RATE[,...]")
    proxy.add_argument("--latency-s", type=float, default=5.0, help="delay of the latency fault")
    proxy.add_argument("--seed", type=int, default=0)
    proxy.add_argument("--out", type=Path, default=Path("llm_calls.jsonl"), help="JSONL event log")
    proxy.set_defaults(handler=_run_proxy)
    return parser


def _options(args: argparse.Namespace) -> RunOptions:
    flags = {k: v for k, v in vars(args).items() if k not in CONTROL}
    resolved = resolve(flags, profile=args.profile, project=load_project(Path.cwd()))
    resolved.setdefault("project", Path.cwd().name)
    return RunOptions(**resolved)  # ty: ignore[invalid-argument-type]  keys come from config.OPTIONS


def _load_baseline(options: RunOptions) -> Baseline | None:
    """Read the baseline when one is given; fail early when a gate needs one and none is."""
    gates = parse_all(options.fail_if)
    if options.baseline is not None:
        return baselines.load(options.baseline)
    if any(gate.uses_baseline for gate in gates):
        raise ConfigError("a --fail-if compares with baseline, but no --baseline was given")
    return None


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


def _kind(config: Path) -> object:
    return json.loads(config.read_text()).get("kind")


def _projection(options: RunOptions) -> str:
    """Executions and spend a timed run will take, from the last preflight's run time and cost."""
    runs = sorted((options.runs_dir / options.project).glob("*/config.json"), reverse=True)
    preflight = next((c.parent for c in runs if _kind(c) == "doctor"), None)
    if preflight is None:
        return "no preflight yet: run `instar doctor` to project this run's spend"
    data = RunData.load(preflight)
    if not data.executions:
        return f"preflight {preflight.name} recorded no execution; cannot project"
    execution = data.executions[0]
    run_s = max(execution.run_ns / 1e9, 1e-3)
    if options.rate_per_s is not None and options.hold_s is not None:
        executions = options.rate_per_s * options.hold_s
    else:
        executions = sum(
            s.users * s.hold_s / (run_s + options.think_time_s)
            for s in options.shape()
            if s.hold_s is not None
        )
    cost = data.cost(execution)
    line = (
        f"projection from preflight {preflight.name}:"
        f" about {executions:.0f} executions of {run_s:.1f} s"
    )
    if cost is None:
        return f"{line}; spend unknown (the preflight reported no cost)"
    spend = executions * cost
    line = f"{line}, about ${spend:.2f}"
    if options.spend_cap_usd is not None and spend > options.spend_cap_usd:
        line += f"\nwarning: the projection exceeds the spend cap of ${options.spend_cap_usd:g}"
    return line


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


def _console(a: Analysis, run_dir: Path) -> None:
    s = a.summary
    statuses = " · ".join(f"{k} {v}" for k, v in sorted(s.statuses.items())) or "none"
    primary = a.metrics.get(a.primary)
    line = f"run {a.run_id}: {s.executions} executions · {statuses}"
    if primary is not None and primary.value is not None:
        if a.primary in a.percent_metrics and primary.low is not None and primary.high is not None:
            line += f" · {a.primary} {primary.value:.0%} ({primary.low:.0%}-{primary.high:.0%})"
        else:
            line += f" · {a.primary} {primary.value:.3g}"
    print(line)
    if a.breaking_point:
        print(f"breaking point: stage {a.breaking_point.stage} ({a.breaking_point.users} users)")
    for gate in a.gates:
        print(f"gate {gate.state}: {gate.text}: {gate.sentence}")
    if a.changes is not None:
        print(f"regressions against baseline: {sum(c.regressed for c in a.changes)}")
    print(f"report: {run_dir / 'report.md'}")


def _run(args: argparse.Namespace) -> int:
    try:
        options = _options(args)
        if args.debug:
            options = copy.replace(
                options,
                executor="inline",
                users=1,
                stages=None,
                hold_s=None,
                rate_per_s=None,
                repeat=1,
                limit=1,
                keep_outputs=True,
            )
        baseline = _load_baseline(options)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if options.timed:
        print(_projection(options), file=sys.stderr)
    result = _start(Runner(options))
    if isinstance(result, int):
        return result
    if args.debug:
        _print_debug(result)
    analysis = analyze(result.run_dir, options, baseline=baseline)
    write_reports(result.run_dir, analysis)
    _console(analysis, result.run_dir)
    if analysis.gate_error:
        print(f"error: {analysis.gate_error}", file=sys.stderr)
        return 2
    if analysis.could_not_run:
        print(f"error: the test could not run: {analysis.could_not_run}", file=sys.stderr)
        return 3
    if result.cancelled:
        print("run cancelled before every task ran", file=sys.stderr)
        return 3
    return gate_exit_code(analysis.gates, strict=options.strict)


def _doctor(args: argparse.Namespace) -> int:
    try:
        options = copy.replace(
            _options(args),
            users=1,
            stages=(Stage(users=1),),
            hold_s=None,
            rate_per_s=None,
            repeat=1,
            limit=1,
            faults={},
            fail_if=(),
            kind="doctor",
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = _start(Runner(options))
    if isinstance(result, int):
        return result
    diagnosis = diagnose(options, result)
    print("\n".join(diagnosis.lines))
    print("doctor: ok" if not diagnosis.problems else f"doctor: {diagnosis.problems} problem(s)")
    return diagnosis.exit_code


def _saved_options(run_dir: Path) -> RunOptions:
    config = json.loads((run_dir / "config.json").read_text())
    return RunOptions(
        seed=config.get("seed", 0),
        fail_if=tuple(config.get("fail_if", ())),
        baseline=Path(config["baseline"]) if config.get("baseline") else None,
        strict=config.get("strict", False),
        interval_s=config.get("interval_s", 30.0),
        breaking_p95=config.get("breaking_p95", 2.0),
    )


def _report(args: argparse.Namespace) -> int:
    if not (args.run_dir / "run.sqlite").exists():
        print(f"error: no run store in {args.run_dir}", file=sys.stderr)
        return 2
    options = _saved_options(args.run_dir)
    try:
        baseline = baselines.load(options.baseline) if options.baseline else None
    except ConfigError as exc:
        print(f"warning: {exc}; reporting without it", file=sys.stderr)
        baseline = None
    print(RENDERERS[args.format](analyze(args.run_dir, options, baseline=baseline)), end="")
    return 0


def _baseline(args: argparse.Namespace) -> int:
    if not (args.run_dir / "run.sqlite").exists():
        print(f"error: no run store in {args.run_dir}", file=sys.stderr)
        return 2
    options = copy.replace(_saved_options(args.run_dir), fail_if=())
    baselines.save(analyze(args.run_dir, options), args.path)
    print(f"baseline saved to {args.path}")
    return 0


def _run_proxy(args: argparse.Namespace) -> int:
    try:
        resolved = resolve({"faults": args.faults} if args.faults else {}, profile=None, project={})
        declared = dict(item.partition("=")[::2] for item in args.upstream)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if any(not name or not url for name, url in declared.items()):
        print("error: --upstream takes NAME=URL", file=sys.stderr)
        return 2
    upstreams = resolve_upstreams(declared)
    with closing(JsonlSink(args.out)) as sink:
        proxy = Proxy(
            upstreams=upstreams,
            sink=sink,
            seed=args.seed,
            fault_rates=resolved.get("faults", {}),  # ty: ignore[invalid-argument-type]  converted by config
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
