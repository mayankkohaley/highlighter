"""Stage 1: expand a raw question into a structured Query."""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.query import Query

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_INSTRUCTIONS = (
    "Expand the user's question into a query plan for a document-reading "
    "assistant. Return: (a) 3 to 7 focused sub-questions or facets that "
    "decompose the original question, and (b) a one-to-two sentence relevance "
    "rubric describing what would count as a useful verbatim excerpt when "
    "answering the question."
)


class _QueryExpansion(BaseModel):
    sub_questions: list[str]
    rubric: str


def build_query_agent(model: str = _DEFAULT_MODEL) -> Agent[None, _QueryExpansion]:
    return Agent(model, output_type=_QueryExpansion, instructions=_INSTRUCTIONS)


def expand_query(
    question: str,
    *,
    agent: Agent[None, _QueryExpansion] | None = None,
) -> Query:
    agent = agent or build_query_agent()
    expansion = agent.run_sync(question).output
    return Query(
        question=question,
        sub_questions=expansion.sub_questions,
        rubric=expansion.rubric,
    )
