"""Stage 0: load and normalize a markdown document."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class NormalizedDoc(BaseModel):
    text: str


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def normalize(path: Path | str) -> NormalizedDoc:
    return NormalizedDoc(text=_normalize_text(Path(path).read_text()))
