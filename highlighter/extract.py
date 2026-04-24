"""Stage 3: extract verbatim excerpts from a chunk that address a query."""
from __future__ import annotations

import re

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.chunk import Chunk
from highlighter.query import Query

# Markdown emphasis/inline-code markers the model may or may not preserve when
# quoting verbatim. Stripped symmetrically from both chunk and candidate for
# the substring verification — keeps legitimate extractions from being dropped
# when the model normalizes away bold/italic/code markers.
_EMPHASIS_MARKERS = re.compile(r"[*_`]")


def _matches(chunk_text: str, candidate: str) -> bool:
    normalized_chunk = _EMPHASIS_MARKERS.sub("", chunk_text)
    normalized_candidate = _EMPHASIS_MARKERS.sub("", candidate)
    return normalized_candidate in normalized_chunk

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_INSTRUCTIONS = (
    "You extract verbatim spans from a document chunk that address the user's "
    "question. Return EVERY span that is useful — if several bullets, sentences, "
    "or clauses are all relevant, return each one separately. Prefer shorter, "
    "focused spans over long paragraphs. Do not paraphrase, do not summarize, "
    "do not invent. If the chunk contains nothing relevant, return an empty list."
)


class RawExcerpt(BaseModel):
    """Pre-verification span returned by the extractor agent."""
    text: str
    which_subquestion: str | None = None
    confidence: float = 1.0


class _ExtractorOutput(BaseModel):
    excerpts: list[RawExcerpt]


class Excerpt(BaseModel):
    """Verified excerpt with citation metadata attached from the source chunk."""
    text: str
    which_subquestion: str | None = None
    confidence: float = 1.0
    line_start: int
    line_end: int
    section_path: list[str]


def build_extractor_agent(
    model: str = _DEFAULT_MODEL,
) -> Agent[None, _ExtractorOutput]:
    return Agent(
        model,
        output_type=_ExtractorOutput,
        instructions=_INSTRUCTIONS,
        model_settings={"temperature": 0.0},
    )


def _build_prompt(chunk_text: str, query: Query) -> str:
    parts = [f"Question: {query.question}"]
    if query.sub_questions:
        parts.append("Sub-questions:\n" + "\n".join(f"- {s}" for s in query.sub_questions))
    if query.rubric:
        parts.append(f"Relevance rubric: {query.rubric}")
    parts.append(f"Chunk:\n{chunk_text}")
    return "\n\n".join(parts)


class ExtractResult(BaseModel):
    """Full output of one extraction call — the prompt sent, raw candidates, verified excerpts."""
    prompt: str
    raw_candidates: list[RawExcerpt]
    verified: list[Excerpt]


def extract_excerpts_verbose(
    chunk: Chunk,
    query: Query,
    *,
    agent: Agent[None, _ExtractorOutput] | None = None,
) -> ExtractResult:
    agent = agent or build_extractor_agent()
    prompt = _build_prompt(chunk.text, query)
    raw = agent.run_sync(prompt).output.excerpts
    verified = [
        Excerpt(
            text=c.text,
            which_subquestion=c.which_subquestion,
            confidence=c.confidence,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            section_path=chunk.section_path,
        )
        for c in raw
        if _matches(chunk.text, c.text)
    ]
    return ExtractResult(prompt=prompt, raw_candidates=raw, verified=verified)


def extract_excerpts(
    chunk: Chunk,
    query: Query,
    *,
    agent: Agent[None, _ExtractorOutput] | None = None,
) -> list[Excerpt]:
    return extract_excerpts_verbose(chunk, query, agent=agent).verified
