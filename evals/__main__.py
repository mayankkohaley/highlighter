"""CLI: run eval cases and print a precision/recall/F1 report.

    uv run python -m evals                    # all cases, one run each
    uv run python -m evals --case <name>      # single case by name
    uv run python -m evals --runs N           # re-run each case N times,
                                              # report min/max/mean F1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic_ai import Agent

from evals.baseline import DEFAULT_TOLERANCE, aggregate, check_regressions
from evals.baseline import load as load_baseline
from evals.baseline import save as save_baseline
from evals.fixtures import EvalCase, load_cases
from evals.runner import CaseResult, run_case
from highlighter.extract import _ExtractorOutput

_DEFAULT_BASELINE_PATH = "evals/baseline.json"


def _format_case_single(result: CaseResult, debug: bool = False) -> str:
    lines: list[str] = []
    s = result.score
    lines.append(result.case.name)
    lines.append(
        f"  precision: {s.precision:.2f}   recall: {s.recall:.2f}   f1: {s.f1:.2f}"
    )
    lines.append(
        f"  predicted: {len(result.predicted)}   "
        f"matched: {len(result.matched_expected)}"
    )
    if result.missing_expected:
        lines.append(f"  missing: {result.missing_expected}")
    if result.predicted:
        lines.append(f"  predicted excerpts ({len(result.predicted)}):")
        for text in result.predicted:
            lines.append(f"    > {text}")
    if debug:
        lines.append("  --- debug ---")
        lines.append(f"  raw candidates ({len(result.raw_candidates)}):")
        for text in result.raw_candidates:
            lines.append(f"    > {text}")
        lines.append("  prompt:")
        for line in result.prompt.splitlines():
            lines.append(f"    | {line}")
    return "\n".join(lines)


def _format_case_multi(runs: list[CaseResult]) -> str:
    name = runs[0].case.name
    n = len(runs)
    f1s = [r.score.f1 for r in runs]
    ps = [r.score.precision for r in runs]
    rs = [r.score.recall for r in runs]
    lines = [
        name,
        f"  f1:        min={min(f1s):.2f} max={max(f1s):.2f} mean={sum(f1s)/n:.2f} ({n} runs)",
        f"  precision: min={min(ps):.2f} max={max(ps):.2f} mean={sum(ps)/n:.2f}",
        f"  recall:    min={min(rs):.2f} max={max(rs):.2f} mean={sum(rs)/n:.2f}",
    ]
    return "\n".join(lines)


def _format_aggregate(
    runs_per_case: list[list[CaseResult]], n_runs: int
) -> str:
    if not runs_per_case:
        return "No cases found."
    mean_per_case_f1 = [
        sum(r.score.f1 for r in runs) / len(runs) for runs in runs_per_case
    ]
    mean_per_case_p = [
        sum(r.score.precision for r in runs) / len(runs) for runs in runs_per_case
    ]
    mean_per_case_r = [
        sum(r.score.recall for r in runs) / len(runs) for runs in runs_per_case
    ]
    n_cases = len(runs_per_case)
    suffix = f" × {n_runs} runs" if n_runs > 1 else ""
    return (
        f"Aggregate ({n_cases} cases{suffix}): "
        f"precision={sum(mean_per_case_p)/n_cases:.2f}   "
        f"recall={sum(mean_per_case_r)/n_cases:.2f}   "
        f"f1={sum(mean_per_case_f1)/n_cases:.2f}"
    )


def _load_cases(cases_dir: Path, name: str | None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for p in sorted(cases_dir.glob("*.yaml")):
        cases.extend(load_cases(p))
    if name is not None:
        cases = [c for c in cases if c.name == name]
    return cases


def _main(
    argv: list[str],
    *,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="evals",
        description="Run excerpt-extraction evals and print a P/R/F1 report.",
    )
    parser.add_argument("--cases-dir", default="evals/fixtures/cases")
    parser.add_argument("--docs-dir", default="evals/fixtures/docs")
    parser.add_argument("--case", help="Run a single case by name.")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Re-run each case N times and report min/max/mean F1.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print the prompt and raw model output for each case.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="After running, write per-case mean scores to the baseline file.",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Compare current scores to the baseline file; exit non-zero on regression.",
    )
    parser.add_argument(
        "--baseline-path",
        default=_DEFAULT_BASELINE_PATH,
        help=f"Baseline file path (default: {_DEFAULT_BASELINE_PATH}).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"F1 drop tolerance for --check-baseline (default: {DEFAULT_TOLERANCE}).",
    )
    args = parser.parse_args(argv[1:])

    cases = _load_cases(Path(args.cases_dir), args.case)
    runs_per_case: list[list[CaseResult]] = []
    for c in cases:
        runs = [
            run_case(c, docs_dir=args.docs_dir, extract_agent=extract_agent)
            for _ in range(args.runs)
        ]
        runs_per_case.append(runs)
        if args.runs == 1:
            print(_format_case_single(runs[0], debug=args.debug))
        else:
            print(_format_case_multi(runs))
        print()
    print(_format_aggregate(runs_per_case, args.runs))

    if args.write_baseline:
        baseline = aggregate(runs_per_case)
        save_baseline(baseline, args.baseline_path)
        print(f"\nWrote baseline to {args.baseline_path} ({len(baseline.cases)} cases).")

    if args.check_baseline:
        baseline = load_baseline(args.baseline_path)
        current = aggregate(runs_per_case)
        regressions = check_regressions(baseline, current, tolerance=args.tolerance)
        if regressions:
            print(f"\nRegressions ({len(regressions)}, tolerance={args.tolerance}):")
            for r in regressions:
                print(
                    f"  {r.case_name}: baseline f1={r.baseline_f1:.2f}  "
                    f"current f1={r.current_f1:.2f}  delta={r.delta:+.2f}"
                )
            return 1
        print(f"\nBaseline OK: no regressions (tolerance={args.tolerance}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
