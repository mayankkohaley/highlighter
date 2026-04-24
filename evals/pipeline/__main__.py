"""CLI: run pipeline-level eval cases and print a precision/recall/F1 report.

    uv run python -m evals.pipeline                    # all cases, one run each
    uv run python -m evals.pipeline --case <name>      # single case by name
    uv run python -m evals.pipeline --runs N           # re-run each case N times
    uv run python -m evals.pipeline --debug            # also show generated
                                                       # sub-questions and rubric
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic_ai import Agent

from evals.baseline import aggregate, check_regressions
from evals.baseline import load as load_baseline
from evals.baseline import save as save_baseline
from evals.pipeline.fixtures import PipelineCase, load_pipeline_cases
from evals.pipeline.runner import PipelineCaseResult, run_pipeline_case
from highlighter.expand import _QueryExpansion
from highlighter.extract import _ExtractorOutput

_DEFAULT_BASELINE_PATH = "evals/baseline-pipeline.json"
_DEFAULT_TOLERANCE = 0.05


def _format_case_single(result: PipelineCaseResult, debug: bool = False) -> str:
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
        lines.append("  --- stage 1 ---")
        lines.append(f"  sub-questions ({len(result.query.sub_questions)}):")
        for sq in result.query.sub_questions:
            lines.append(f"    - {sq}")
        lines.append(f"  rubric: {result.query.rubric}")
    return "\n".join(lines)


def _format_case_multi(runs: list[PipelineCaseResult]) -> str:
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
    runs_per_case: list[list[PipelineCaseResult]], n_runs: int
) -> str:
    if not runs_per_case:
        return "No cases found."
    mean_p = [sum(r.score.precision for r in runs) / len(runs) for runs in runs_per_case]
    mean_r = [sum(r.score.recall for r in runs) / len(runs) for runs in runs_per_case]
    mean_f1 = [sum(r.score.f1 for r in runs) / len(runs) for runs in runs_per_case]
    n_cases = len(runs_per_case)
    suffix = f" × {n_runs} runs" if n_runs > 1 else ""
    return (
        f"Aggregate ({n_cases} cases{suffix}): "
        f"precision={sum(mean_p)/n_cases:.2f}   "
        f"recall={sum(mean_r)/n_cases:.2f}   "
        f"f1={sum(mean_f1)/n_cases:.2f}"
    )


def _load_cases(cases_dir: Path, name: str | None) -> list[PipelineCase]:
    cases: list[PipelineCase] = []
    for p in sorted(cases_dir.glob("*.yaml")):
        cases.extend(load_pipeline_cases(p))
    if name is not None:
        cases = [c for c in cases if c.name == name]
    return cases


def _main(
    argv: list[str],
    *,
    expand_agent: Agent[None, _QueryExpansion] | None = None,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.pipeline",
        description="Run pipeline-level evals (full Stage 0→3 end-to-end).",
    )
    parser.add_argument("--cases-dir", default="evals/fixtures/pipeline")
    parser.add_argument("--docs-dir", default="evals/fixtures/docs")
    parser.add_argument("--case", help="Run a single case by name.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also print generated sub-questions and rubric for each case.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="After running, write per-case mean scores to the baseline file.",
    )
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Compare current scores to the baseline; exit non-zero on regression.",
    )
    parser.add_argument(
        "--baseline-path",
        default=_DEFAULT_BASELINE_PATH,
        help=f"Baseline file path (default: {_DEFAULT_BASELINE_PATH}).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=_DEFAULT_TOLERANCE,
        help=f"F1 drop tolerance for --check-baseline (default: {_DEFAULT_TOLERANCE}).",
    )
    args = parser.parse_args(argv[1:])

    cases = _load_cases(Path(args.cases_dir), args.case)
    runs_per_case: list[list[PipelineCaseResult]] = []
    for c in cases:
        runs = [
            run_pipeline_case(
                c,
                docs_dir=args.docs_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                expand_agent=expand_agent,
                extract_agent=extract_agent,
            )
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
