"""Run an eval case: load doc, select chunk, extract, score."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from evals.fixtures import EvalCase
from evals.score import CaseScore, score_case
from evals.selector import select_chunk
from highlighter.extract import _ExtractorOutput, extract_excerpts_verbose
from highlighter.normalize import normalize


class CaseResult(BaseModel):
    case: EvalCase
    score: CaseScore
    predicted: list[str]
    matched_expected: list[str]
    missing_expected: list[str]
    prompt: str = ""
    raw_candidates: list[str] = []


def run_case(
    case: EvalCase,
    *,
    docs_dir: Path | str,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
) -> CaseResult:
    doc = normalize(Path(docs_dir) / case.document)
    chunk = select_chunk(doc, case.chunk_selector)
    er = extract_excerpts_verbose(chunk, case.query, agent=extract_agent)
    predicted = [e.text for e in er.verified]
    score = score_case(predicted=predicted, expected=case.expected_excerpts)
    matched = [e for e in case.expected_excerpts if any(e in p for p in predicted)]
    missing = [e for e in case.expected_excerpts if e not in matched]
    return CaseResult(
        case=case,
        score=score,
        predicted=predicted,
        matched_expected=matched,
        missing_expected=missing,
        prompt=er.prompt,
        raw_candidates=[c.text for c in er.raw_candidates],
    )
