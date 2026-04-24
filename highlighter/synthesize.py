"""Stage 5: synthesize a final answer from consolidated excerpts.

The only step that sees the whole picture. Takes the original question plus
the consolidated excerpts with their citations and produces a short answer
that references the excerpts by bracketed index (`[1]`, `[2]`, ...).
"""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from highlighter.extract import Excerpt
from highlighter.query import Query

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_INSTRUCTIONS = (
    "Answer the user's question using ONLY the numbered excerpts provided. "
    "Cite evidence by bracketed excerpt number, e.g. `[1]` or `[2, 3]`. Do "
    "not introduce facts outside the excerpts. If the excerpts do not answer "
    "the question, say so plainly. Populate `used_excerpts` with the 1-indexed "
    "positions you actually cited."
)


class Synthesis(BaseModel):
    answer: str
    used_excerpts: list[int] = []


def build_synthesis_agent(model: str = _DEFAULT_MODEL) -> Agent[None, Synthesis]:
    return Agent(
        model,
        output_type=Synthesis,
        instructions=_INSTRUCTIONS,
        model_settings={"temperature": 0.0},
    )


def _format_excerpt(i: int, e: Excerpt) -> str:
    path = " > ".join(e.section_path) if e.section_path else "(no heading)"
    return f"[{i}] {path}  L{e.line_start}-{e.line_end}\n{e.text}"


def _build_prompt(query: Query, excerpts: list[Excerpt]) -> str:
    body = "\n\n".join(
        _format_excerpt(i, e) for i, e in enumerate(excerpts, start=1)
    )
    return f"Question: {query.question}\n\nExcerpts:\n\n{body}"


def synthesize(
    query: Query,
    excerpts: list[Excerpt],
    *,
    agent: Agent[None, Synthesis] | None = None,
) -> Synthesis:
    if not excerpts:
        return Synthesis(
            answer="No relevant excerpts were found for this question.",
            used_excerpts=[],
        )
    agent = agent or build_synthesis_agent()
    return agent.run_sync(_build_prompt(query, excerpts)).output
