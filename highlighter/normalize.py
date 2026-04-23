"""Stage 0: load and normalize a markdown document."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel


class NormalizedDoc(BaseModel):
    source_path: str
    content_hash: str
    text: str


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def normalize(path: Path | str) -> NormalizedDoc:
    path = Path(path)
    raw = path.read_bytes()
    return NormalizedDoc(
        source_path=str(path),
        content_hash=hashlib.sha256(raw).hexdigest(),
        text=_normalize_text(raw.decode("utf-8")),
    )
