"""Load pipeline-level eval cases.

Per-doc YAML fixture shape:

    document: <filename>        # in evals/fixtures/docs/
    cases:
      - name: <id>
        question: <raw user question>
        expected_excerpts:
          - <verbatim phrase 1>
          - <verbatim phrase 2>
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class PipelineCase(BaseModel):
    name: str
    document: str
    question: str
    expected_excerpts: list[str]


class _FileCase(BaseModel):
    name: str
    question: str
    expected_excerpts: list[str]


class _FixtureFile(BaseModel):
    document: str
    cases: list[_FileCase]


def load_pipeline_cases(path: Path | str) -> list[PipelineCase]:
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return []
    fixture = _FixtureFile.model_validate(data)
    return [
        PipelineCase(
            name=c.name,
            document=fixture.document,
            question=c.question,
            expected_excerpts=c.expected_excerpts,
        )
        for c in fixture.cases
    ]
