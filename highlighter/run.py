"""End-to-end pipeline: normalize → expand → chunk → extract."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.chunk import Chunk, chunk_document
from highlighter.consolidate import consolidate
from highlighter.expand import _QueryExpansion, expand_query
from highlighter.extract import (
    Excerpt,
    ExtractResult,
    RawExcerpt,
    _ExtractorOutput,
    extract_excerpts_verbose_async,
)
from highlighter.normalize import NormalizedDoc, normalize
from highlighter.query import Query
from highlighter.synthesize import Synthesis
from highlighter.synthesize import synthesize as _synthesize


class PipelineResult(BaseModel):
    query: Query
    excerpts: list[Excerpt]
    consolidated: list[Excerpt] = []
    raw_candidates: list[RawExcerpt] = []
    synthesis: Synthesis | None = None


_CONCURRENCY_CEILING = 8


async def _extract_all(
    chunks: list[Chunk],
    query: Query,
    doc: NormalizedDoc,
    extract_agent: Agent[None, _ExtractorOutput] | None,
) -> list[ExtractResult]:
    # Cap at min(chunks, ceiling): no reason to stand up semaphore slots that
    # can never be filled, and capping at 8 keeps us well under Anthropic's
    # per-minute concurrency limits for Haiku.
    semaphore = asyncio.Semaphore(min(len(chunks), _CONCURRENCY_CEILING) or 1)

    async def _one(chunk: Chunk) -> ExtractResult:
        async with semaphore:
            return await extract_excerpts_verbose_async(
                chunk, query, doc, agent=extract_agent
            )

    raw = await asyncio.gather(
        *(_one(c) for c in chunks), return_exceptions=True
    )
    results: list[ExtractResult] = []
    for chunk, item in zip(chunks, raw, strict=True):
        if isinstance(item, BaseException):
            print(
                f"[highlighter] per-chunk extraction failed at "
                f"L{chunk.line_start}-L{chunk.line_end}: {item!r}",
                file=sys.stderr,
            )
            results.append(ExtractResult(prompt="", raw_candidates=[], verified=[]))
        else:
            results.append(item)
    return results


def run_pipeline(
    doc_path: str | Path,
    question: str,
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    synthesize: bool = False,
    expand_agent: Agent[None, _QueryExpansion] | None = None,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
    synthesis_agent: Agent[None, Synthesis] | None = None,
) -> PipelineResult:
    doc = normalize(doc_path)
    query = expand_query(question, agent=expand_agent)
    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    results = asyncio.run(_extract_all(chunks, query, doc, extract_agent))
    excerpts: list[Excerpt] = []
    raw_candidates: list[RawExcerpt] = []
    for er in results:
        excerpts.extend(er.verified)
        raw_candidates.extend(er.raw_candidates)
    consolidated = consolidate(excerpts, doc)
    answer = _synthesize(query, consolidated, agent=synthesis_agent) if synthesize else None
    return PipelineResult(
        query=query,
        excerpts=excerpts,
        consolidated=consolidated,
        raw_candidates=raw_candidates,
        synthesis=answer,
    )
