"""CLI: run eval cases and print a precision/recall/F1 report.

    uv run python -m evals                    # all cases in evals/fixtures/cases/
    uv run python -m evals --case <name>      # single case by name
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic_ai import Agent

from evals.fixtures import EvalCase, load_cases
from evals.runner import CaseResult, run_case
from highlighter.extract import _ExtractorOutput


def _format_case(result: CaseResult) -> str:
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
    return "\n".join(lines)


def _format_aggregate(results: list[CaseResult]) -> str:
    if not results:
        return "No cases found."
    n = len(results)
    p = sum(r.score.precision for r in results) / n
    r = sum(r.score.recall for r in results) / n
    f = sum(r.score.f1 for r in results) / n
    return f"Aggregate ({n} cases): precision={p:.2f}   recall={r:.2f}   f1={f:.2f}"


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
    args = parser.parse_args(argv[1:])

    cases = _load_cases(Path(args.cases_dir), args.case)
    results = [
        run_case(c, docs_dir=args.docs_dir, extract_agent=extract_agent) for c in cases
    ]
    for r in results:
        print(_format_case(r))
        print()
    print(_format_aggregate(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
