"""CLI: run the full highlighter pipeline over a markdown file.

    uv run python -m highlighter <file> -q "your question"
"""
from __future__ import annotations

import argparse
import sys

from pydantic_ai import Agent

from highlighter.expand import _QueryExpansion
from highlighter.extract import Excerpt, _ExtractorOutput
from highlighter.query import Query
from highlighter.run import run_pipeline


def _format_result(query: Query, excerpts: list[Excerpt]) -> str:
    lines: list[str] = []
    lines.append(f"Question: {query.question}")
    lines.append("")
    if query.sub_questions:
        lines.append("Sub-questions:")
        for sq in query.sub_questions:
            lines.append(f"  - {sq}")
        lines.append("")
    if query.rubric:
        lines.append(f"Rubric: {query.rubric}")
        lines.append("")
    lines.append(f"Excerpts ({len(excerpts)}):")
    lines.append("")
    for i, e in enumerate(excerpts, start=1):
        path = " > ".join(e.section_path) if e.section_path else "(no heading)"
        lines.append(f"[{i}] L{e.line_start}-L{e.line_end}  {path}")
        meta: list[str] = []
        if e.which_subquestion:
            meta.append(f"sub-q: {e.which_subquestion}")
        meta.append(f"confidence: {e.confidence:.2f}")
        lines.append("    " + "   ".join(meta))
        lines.append(f"    > {e.text}")
        lines.append("")
    return "\n".join(lines)


def _main(
    argv: list[str],
    *,
    expand_agent: Agent[None, _QueryExpansion] | None = None,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="highlighter",
        description=(
            "Ask a question about a markdown file and get verbatim excerpts with citations."
        ),
    )
    parser.add_argument("file", help="Path to a markdown document.")
    parser.add_argument("-q", "--question", required=True, help="The question to ask.")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args(argv[1:])

    result = run_pipeline(
        args.file,
        args.question,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        expand_agent=expand_agent,
        extract_agent=extract_agent,
    )
    print(_format_result(result.query, result.excerpts))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
