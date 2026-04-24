"""Stage 3: extract verbatim excerpts from a chunk that address a query."""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.chunk import Chunk
from highlighter.matching import find_span
from highlighter.normalize import NormalizedDoc
from highlighter.query import Query

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_INSTRUCTIONS = (
    "You extract verbatim spans from a document chunk that address the user's "
    "question. Return EVERY span that is useful — if several bullets, sentences, "
    "or clauses are all relevant, return each one separately. Prefer the natural "
    "unit of meaning (full bullet, full sentence, discrete entry) — not a whole "
    "paragraph, not a mid-unit fragment. Preserve any PR numbers, issue links, "
    "citations, or attributions attached to the span verbatim. Do not paraphrase, "
    "do not summarize, do not invent. If the chunk contains nothing relevant, "
    "return an empty list."
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


def _verify_candidates(
    chunk: Chunk,
    doc: NormalizedDoc,
    raw: list[RawExcerpt],
) -> list[Excerpt]:
    verified: list[Excerpt] = []
    for c in raw:
        span = find_span(chunk.text, c.text)
        if span is None:
            continue
        start, end = span
        line_start = chunk.line_start + chunk.text.count("\n", 0, start)
        line_end = chunk.line_start + chunk.text.count("\n", 0, end - 1)
        verified.append(Excerpt(
            text=c.text,
            which_subquestion=c.which_subquestion,
            confidence=c.confidence,
            line_start=line_start,
            line_end=line_end,
            section_path=doc.section_path_for_line(line_start),
        ))
    return verified


def extract_excerpts_verbose(
    chunk: Chunk,
    query: Query,
    doc: NormalizedDoc,
    *,
    agent: Agent[None, _ExtractorOutput] | None = None,
) -> ExtractResult:
    agent = agent or build_extractor_agent()
    prompt = _build_prompt(chunk.text, query)
    raw = agent.run_sync(prompt).output.excerpts
    return ExtractResult(
        prompt=prompt,
        raw_candidates=raw,
        verified=_verify_candidates(chunk, doc, raw),
    )


async def extract_excerpts_verbose_async(
    chunk: Chunk,
    query: Query,
    doc: NormalizedDoc,
    *,
    agent: Agent[None, _ExtractorOutput] | None = None,
) -> ExtractResult:
    agent = agent or build_extractor_agent()
    prompt = _build_prompt(chunk.text, query)
    raw = (await agent.run(prompt)).output.excerpts
    return ExtractResult(
        prompt=prompt,
        raw_candidates=raw,
        verified=_verify_candidates(chunk, doc, raw),
    )


def extract_excerpts(
    chunk: Chunk,
    query: Query,
    doc: NormalizedDoc,
    *,
    agent: Agent[None, _ExtractorOutput] | None = None,
) -> list[Excerpt]:
    return extract_excerpts_verbose(chunk, query, doc, agent=agent).verified
