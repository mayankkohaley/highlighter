"""Stage 3: extract verbatim excerpts from a chunk that address a query."""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.chunk import Chunk
from highlighter.query import Query

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_INSTRUCTIONS = (
    "You extract verbatim spans from a document chunk that address the user's "
    "question. Do not paraphrase, do not summarize, do not invent. If the chunk "
    "contains nothing relevant, return an empty list."
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


def build_extractor_agent(
    model: str = _DEFAULT_MODEL,
) -> Agent[None, _ExtractorOutput]:
    return Agent(model, output_type=_ExtractorOutput, instructions=_INSTRUCTIONS)


def _build_prompt(chunk_text: str, query: Query) -> str:
    parts = [f"Question: {query.question}"]
    if query.sub_questions:
        parts.append("Sub-questions:\n" + "\n".join(f"- {s}" for s in query.sub_questions))
    if query.rubric:
        parts.append(f"Relevance rubric: {query.rubric}")
    parts.append(f"Chunk:\n{chunk_text}")
    return "\n\n".join(parts)


def extract_excerpts(
    chunk: Chunk,
    query: Query,
    *,
    agent: Agent[None, _ExtractorOutput] | None = None,
) -> list[Excerpt]:
    agent = agent or build_extractor_agent()
    prompt = _build_prompt(chunk.text, query)
    result = agent.run_sync(prompt)
    return [
        Excerpt(text=c.text)
        for c in result.output.excerpts
        if c.text in chunk.text
    ]
