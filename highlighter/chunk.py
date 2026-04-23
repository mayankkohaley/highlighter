"""Stage 2: chunk a normalized document into overlapping, boundary-aware pieces."""
from __future__ import annotations

from chonkie import RecursiveChunker
from pydantic import BaseModel

from highlighter.normalize import NormalizedDoc


class Chunk(BaseModel):
    text: str
    line_start: int
    line_end: int


def _line_number_at(text: str, char_index: int) -> int:
    """Return 1-indexed line number of `char_index` in `text`."""
    return text.count("\n", 0, char_index) + 1


def chunk_document(
    doc: NormalizedDoc,
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    chunker = RecursiveChunker.from_recipe(
        "markdown", lang="en", tokenizer="cl100k_base", chunk_size=chunk_size
    )
    raw_chunks = chunker(doc.text)
    return [
        Chunk(
            text=c.text,
            line_start=_line_number_at(doc.text, c.start_index),
            line_end=_line_number_at(doc.text, max(c.end_index - 1, c.start_index)),
        )
        for c in raw_chunks
    ]
