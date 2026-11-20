from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn

from instar.proxy.app import KEEP_ALIVE_S, Proxy
from instar.proxy.faults import check_rates
from instar.proxy.upstreams import resolve_upstreams
from instar.store import JsonlSink

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instar", description="Pre-release testing for AI agents."
    )
    commands = parser.add_subparsers(dest="command", required=True)

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
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
