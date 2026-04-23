"""Stage 2: chunk a normalized document into overlapping, boundary-aware pieces."""
from __future__ import annotations

from chonkie import RecursiveChunker
from pydantic import BaseModel

from highlighter.normalize import NormalizedDoc


class Chunk(BaseModel):
    text: str


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
    return [Chunk(text=c.text) for c in raw_chunks]
