from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from instar.analyze.gate import GateState

if TYPE_CHECKING:
    from instar.analyze import Analysis
    from instar.analyze.curve import Group
    from instar.analyze.stats import Estimate

ICONS = {GateState.PASSED: "✅", GateState.FAILED: "❌", GateState.INCONCLUSIVE: "⚠️"}


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _estimate(estimate: Estimate | None, percent: bool) -> str:
    if estimate is None or estimate.value is None:
        return "n/a"
    show = (lambda v: f"{v:.0%}") if percent else (lambda v: f"{v:.3g}")
    if estimate.low is None or estimate.high is None:
        return show(estimate.value)
    return f"{show(estimate.value)} ({show(estimate.low)}-{show(estimate.high)})"


def _statuses(counts: dict[str, int]) -> str:
    return " · ".join(f"{status} {n}" for status, n in sorted(counts.items())) or "none"


def _group_line(name: str, group: Group, primary: str, percent: bool) -> str:
    return (
        f"| {name} | {group.executions} | {_estimate(group.primary, percent)} |"
        f" {_seconds(group.p50_run_s)} | {_usd(group.cost_per_pass)} |"
        f" {_statuses(group.statuses)} |"
    )


def markdown(a: Analysis) -> str:
    percent = a.primary in a.percent_metrics
    s = a.summary
    lines = [f"# instar run `{a.run_id}` ({a.project})", ""]
    if a.could_not_run:
        lines += [f"**The test could not run:** {a.could_not_run}", ""]
    lines += [
        f"{s.executions} executions: {_statuses(s.statuses)}",
        "",
        f"**{a.primary}: {_estimate(a.metrics.get(a.primary), percent)}** · p50 run"
        f" {_seconds(s.p50_run_s)} · p95 run {_seconds(s.p95_run_s)} · cost per passed task"
        f" {_usd(s.cost_per_pass)} · model calls {s.model_calls}, errors {s.model_errors},"
        f" faults {s.faults}",
        "",
    ]
    if a.gate_error:
        lines += [f"**The gates could not be checked:** {a.gate_error}", ""]
    if a.gates:
        lines += ["## Gates", ""]
        lines += [f"- {ICONS[g.state]} `{g.text}`: {g.sentence}" for g in a.gates]
        lines.append("")
    if len(a.stages) > 1 or any(stage.hold_s is not None for stage in a.stages):
        lines += [
            "## Load",
            "",
            f"| Stage | Users | Hold | Steady executions | {a.primary} | p50 | p95 |"
            " Model errors | Faults | Tasks/h | Shed |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for st in a.stages:
            g = st.steady
            hold = "until done" if st.hold_s is None else f"{st.hold_s:g} s"
            per_hour = "n/a" if st.tasks_per_hour is None else f"{st.tasks_per_hour:.0f}"
            shed = "n/a" if st.arrivals is None else f"{st.shed}/{st.arrivals}"
            lines.append(
                f"| {st.stage} | {st.users} | {hold} | {g.executions} |"
                f" {_estimate(g.primary, percent)} | {_seconds(g.p50_run_s)} |"
                f" {_seconds(g.p95_run_s)} | {g.model_errors} | {g.faults} | {per_hour} | {shed} |"
            )
        lines.append("")
        if a.breaking_point:
            bp = a.breaking_point
            lines.append(f"**Breaking point: stage {bp.stage} ({bp.users} users).** {bp.reason}.")
        else:
            lines.append("No breaking point in these stages.")
        lines.append("")
    if a.repeats is not None:
        lines += [
            "## First attempts and repeats",
            "",
            "Repeats can hit provider caches and look cheaper and faster.",
            "",
            f"| | Executions | {a.primary} | p50 run | Cost per pass | Statuses |",
            "|---|---|---|---|---|---|",
            _group_line("First attempts", a.first_attempts, a.primary, percent),
            _group_line("Repeats", a.repeats, a.primary, percent),
            "",
        ]
    if a.changes is not None:
        regressed = [c for c in a.changes if c.regressed]
        lines += [f"## Compared with baseline `{a.baseline_run}`", ""]
        if regressed:
            lines += [f"{len(regressed)} tasks regressed:", ""]
            for change in regressed:
                before = _maybe_pct(change.before, percent)
                lines.append(
                    f"- `{change.task_id}`: {before} -> {_maybe_pct(change.after, percent)}"
                )
        else:
            lines.append("No task regressed.")
        lines.append("")
    lines += [
        "## Tasks",
        "",
        f"| Task | Executions | {a.primary} | Statuses | Cost |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| `{row.task_id}` | {row.executions} | {_maybe_pct(row.primary, percent)} |"
        f" {_statuses(row.statuses)} | {_usd(row.cost_usd)} |"
        for row in a.tasks
    ]
    return "\n".join(lines) + "\n"


def _maybe_pct(value: float | None, percent: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if percent else f"{value:.3g}"


def to_json(a: Analysis) -> str:
    document = asdict(a)
    document["gates"] = [{**asdict(g), "sentence": g.sentence} for g in a.gates]
    document["percent_metrics"] = sorted(a.percent_metrics)
    if a.changes is not None:
        document["changes"] = [{**asdict(c), "regressed": c.regressed} for c in a.changes]
    return json.dumps(document, indent=2, default=str) + "\n"


def junit(a: Analysis) -> str:
    """One test case per task, plus one per gate in a second suite."""
    suites = ET.Element("testsuites", name=f"instar {a.run_id}")
    tasks = ET.SubElement(
        suites,
        "testsuite",
        name=f"instar.{a.project}",
        tests=str(len(a.tasks)),
        failures=str(sum(row.failing for row in a.tasks)),
    )
    for row in a.tasks:
        case = ET.SubElement(tasks, "testcase", classname=a.project, name=row.task_id)
        if row.failing:
            detail = f"{a.primary} {_maybe_pct(row.primary, True)}; {_statuses(row.statuses)}"
            ET.SubElement(case, "failure", message=detail)
    if a.gates:
        failed = [g for g in a.gates if g.state is GateState.FAILED]
        gates = ET.SubElement(
            suites,
            "testsuite",
            name="instar.gates",
            tests=str(len(a.gates)),
            failures=str(len(failed)),
        )
        for g in a.gates:
            case = ET.SubElement(gates, "testcase", classname="gates", name=g.text)
            if g.state is GateState.FAILED:
                ET.SubElement(case, "failure", message=g.sentence)
            elif g.state is GateState.INCONCLUSIVE:
                ET.SubElement(case, "system-out").text = g.sentence
    ET.indent(suites)
    return ET.tostring(suites, encoding="unicode", xml_declaration=True) + "\n"


RENDERERS = {"md": markdown, "json": to_json, "junit": junit}


def write_reports(run_dir: Path, a: Analysis) -> None:
    text = markdown(a)
    (run_dir / "report.md").write_text(text)
    (run_dir / "report.json").write_text(to_json(a))
    (run_dir / "junit.xml").write_text(junit(a))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as file:
            file.write(text)
