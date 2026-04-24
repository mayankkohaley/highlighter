"""End-to-end pipeline: normalize → expand → chunk → extract."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.chunk import chunk_document
from highlighter.expand import _QueryExpansion, expand_query
from highlighter.extract import Excerpt, _ExtractorOutput, extract_excerpts
from highlighter.normalize import normalize
from highlighter.query import Query


class PipelineResult(BaseModel):
    query: Query
    excerpts: list[Excerpt]


def run_pipeline(
    doc_path: str | Path,
    question: str,
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    expand_agent: Agent[None, _QueryExpansion] | None = None,
    extract_agent: Agent[None, _ExtractorOutput] | None = None,
) -> PipelineResult:
    doc = normalize(doc_path)
    query = expand_query(question, agent=expand_agent)
    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    excerpts: list[Excerpt] = []
    for chunk in chunks:
        excerpts.extend(extract_excerpts(chunk, query, doc, agent=extract_agent))
    return PipelineResult(query=query, excerpts=excerpts)
