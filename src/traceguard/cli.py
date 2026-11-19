"""Command-line interface for TraceGuard benchmark, fuzzing, HTML reporting, and server."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import uvicorn

from traceguard.contracts import FaultMix, PolicyBenchmarkRow
from traceguard.evaluator import (
    evaluate_policies,
    format_leaderboard_markdown,
    get_builtin_scenarios,
    render_html_report,
)
from traceguard.fuzzer import TraceFuzzer


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="traceguard-eval",
        description="Multi-Step AI Agent Trajectory Simulator, Prompt-Injection & Fault Fuzzer, and Trace Replay Studio.",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run the 12-scenario evaluation suite.")
    parser.add_argument("--fuzz", action="store_true", help="Run a seeded adversarial fault & injection fuzz campaign.")
    parser.add_argument("--runs", type=int, default=30, help="Number of fuzzed scenarios to generate (default: 30).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed for fuzzing (default: 42).")
    parser.add_argument("--injection-prob", type=float, default=0.35, help="Prompt injection probability [0..1].")
    parser.add_argument("--transient-prob", type=float, default=0.40, help="Transient fault probability [0..1].")
    parser.add_argument("--persistent-prob", type=float, default=0.15, help="Persistent outage probability [0..1].")
    parser.add_argument("--malformed-prob", type=float, default=0.20, help="Malformed JSON probability [0..1].")
    parser.add_argument("--dirty-prob", type=float, default=0.25, help="Dirty input schema probability [0..1].")
    parser.add_argument("--policies", nargs="+", default=None, help="Subset of agent policy names to evaluate.")
    parser.add_argument(
        "--export-html",
        nargs="?",
        const="traceguard-report.html",
        default=None,
        help="Write standalone HTML trajectory audit report to file.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print structured JSON output.")
    parser.add_argument("--serve", action="store_true", help="Launch the FastAPI Trace Replay Studio web server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for --serve (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for --serve (default: 8000).")
    parser.add_argument(
        "--min-composite",
        type=float,
        default=None,
        help=(
            "Exit non-zero when no evaluated policy reaches this composite score "
            "in [0, 100], so the benchmark can gate a CI job."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.serve:
        uvicorn.run("traceguard.server:app", host=args.host, port=args.port, log_level="info")
        return 0

    run_bench = args.benchmark or (not args.fuzz)
    output_payload: dict[str, Any] = {}
    last_leaderboard = []
    all_rows: list[PolicyBenchmarkRow] = []
    last_trajectories = []
    report_meta: dict[str, Any] = {}
    report_title = "TraceGuard Trajectory Audit Report"

    if run_bench:
        scenarios = get_builtin_scenarios()
        leaderboard, trajectories = evaluate_policies(scenarios, args.policies)
        last_leaderboard = leaderboard
        all_rows.extend(leaderboard)
        last_trajectories = trajectories
        report_title = "TraceGuard 12-Scenario Suite Audit Report"
        report_meta = {"mode": "benchmark", "scenarios": len(scenarios)}
        output_payload["benchmark"] = {
            "scenario_count": len(scenarios),
            "leaderboard": [r.model_dump() for r in leaderboard],
            "trajectories": [t.model_dump() for t in trajectories],
        }
        if not args.as_json:
            print("\n=== TraceGuard 12-Scenario Evaluation Benchmark ===")
            print(format_leaderboard_markdown(leaderboard))

    if args.fuzz:
        mix = FaultMix(
            runs=args.runs,
            seed=args.seed,
            injection_prob=args.injection_prob,
            transient_fault_prob=args.transient_prob,
            persistent_fault_prob=args.persistent_prob,
            malformed_json_prob=args.malformed_prob,
            dirty_arg_prob=args.dirty_prob,
        )
        tasks = TraceFuzzer(mix).generate()
        breakdown = TraceFuzzer.summarize_mix(tasks)
        leaderboard, trajectories = evaluate_policies(tasks, args.policies)
        last_leaderboard = leaderboard
        all_rows.extend(leaderboard)
        last_trajectories = trajectories
        report_title = f"TraceGuard Adversarial Fuzz Report (runs={mix.runs}, seed={mix.seed})"
        report_meta = {"mode": "fuzz", "runs": mix.runs, "seed": mix.seed}
        output_payload["fuzz"] = {
            "fault_mix": mix.model_dump(),
            "breakdown": breakdown,
            "leaderboard": [r.model_dump() for r in leaderboard],
            "trajectories": [t.model_dump() for t in trajectories],
        }
        if not args.as_json:
            print(f"\n=== TraceGuard Adversarial Fuzz Campaign (runs={mix.runs}, seed={mix.seed}) ===")
            print(
                f"Generated {breakdown['total_scenarios']} scenarios | "
                f"Faults: {breakdown['fault_counts']} | Dirty Args: {breakdown['dirty_schema_scenarios']}"
            )
            print(format_leaderboard_markdown(leaderboard))

    if args.export_html:
        out_path = Path(args.export_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html_doc = render_html_report(
            title=report_title,
            leaderboard=last_leaderboard,
            trajectories=last_trajectories,
            metadata=report_meta,
        )
        out_path.write_text(html_doc, encoding="utf-8")
        output_payload["html_report"] = str(out_path)
        if not args.as_json:
            print(f"\nExported standalone HTML audit report to: {out_path}")

    if args.as_json:
        print(json.dumps(output_payload, indent=2))

    return exit_code_for(all_rows, args.min_composite)


def exit_code_for(
    rows: Sequence[PolicyBenchmarkRow], min_composite: Optional[float]
) -> int:
    """Map benchmark rows to a process exit code under a score gate.

    Returns 0 when no threshold is given or some evaluated policy reaches it,
    and 1 when the best composite score is below the threshold or no policies
    were evaluated. With ``--policies`` the gate applies to the named policies
    only. The threshold must be a finite value in [0, 100]; anything else raises
    ``ValueError`` so a typo cannot silently disable gating.
    """
    if min_composite is None:
        return 0
    if not math.isfinite(min_composite) or not 0.0 <= min_composite <= 100.0:
        raise ValueError(
            "min_composite must be a finite value in [0, 100]; "
            f"got {min_composite!r}"
        )
    if not rows:
        return 1
    return 1 if max(row.composite_score for row in rows) < min_composite else 0


if __name__ == "__main__":
    raise SystemExit(main())
