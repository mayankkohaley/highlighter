"""Load and validate eval-case YAML fixtures."""
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


def load_case(path: Path | str) -> EvalCase:
    data = yaml.safe_load(Path(path).read_text())
    return EvalCase.model_validate(data)
