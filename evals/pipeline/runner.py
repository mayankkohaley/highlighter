"""Run a pipeline-level eval case: full Stage 0→3 end-to-end, then score."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from evals.pipeline.fixtures import PipelineCase
from evals.score import CaseScore, score_case
from highlighter.expand import _QueryExpansion
from highlighter.extract import _ExtractorOutput
from highlighter.matching import contains
from highlighter.query import Query
from highlighter.run import run_pipeline


class PipelineCaseResult(BaseModel):
    case: PipelineCase
    score: CaseScore
    query: Query
    predicted: list[str]
    matched_expected: list[str]
    missing_expected: list[str]
    raw_candidates: list[str] = []


def run_pipeline_case(
    case: PipelineCase,
    *,
    docs_dir: Path | str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    expand_agent: Agent[None, _QueryExpansion] | None = None,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
) -> PipelineCaseResult:
    result = run_pipeline(
        Path(docs_dir) / case.document,
        case.question,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        expand_agent=expand_agent,
        extract_agent=extract_agent,
    )
    predicted = [e.text for e in result.excerpts]
    score = score_case(predicted=predicted, expected=case.expected_excerpts)
    matched = [
        e for e in case.expected_excerpts if any(contains(p, e) for p in predicted)
    ]
    missing = [e for e in case.expected_excerpts if e not in matched]
    return PipelineCaseResult(
        case=case,
        score=score,
        query=result.query,
        predicted=predicted,
        matched_expected=matched,
        missing_expected=missing,
        raw_candidates=[c.text for c in result.raw_candidates],
    )
