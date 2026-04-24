"""Load and validate eval-case YAML fixtures.

A fixture file holds one document and a list of cases against it:

    document: <filename>     # in evals/fixtures/docs/
    cases:
      - name: <id>
        chunk_selector: ...
        query: ...
        expected_excerpts: [...]
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from highlighter.query import Query


class ChunkSelector(BaseModel):
    section_path: list[str]


class EvalCase(BaseModel):
    name: str
    document: str
    chunk_selector: ChunkSelector
    query: Query
    expected_excerpts: list[str]


class _FileCase(BaseModel):
    """The per-case shape inside a fixture file (no document — that's shared)."""
    name: str
    chunk_selector: ChunkSelector
    query: Query
    expected_excerpts: list[str]


class _FixtureFile(BaseModel):
    document: str
    cases: list[_FileCase]


def load_cases(path: Path | str) -> list[EvalCase]:
    """Load all cases from a fixture file, stamping the shared `document` onto each.

    A file containing only comments (or empty) is treated as "no cases" — handy
    for commenting out a WIP fixture without deleting it.
    """
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return []
    fixture = _FixtureFile.model_validate(data)
    return [
        EvalCase(
            name=c.name,
            document=fixture.document,
            chunk_selector=c.chunk_selector,
            query=c.query,
            expected_excerpts=c.expected_excerpts,
        )
        for c in fixture.cases
    ]
