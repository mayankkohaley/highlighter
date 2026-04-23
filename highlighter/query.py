"""Stage 1 artifact: the expanded query passed to extraction."""
from __future__ import annotations

from pydantic import BaseModel


class Query(BaseModel):
    question: str
    sub_questions: list[str] = []
    rubric: str = ""
