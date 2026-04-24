"""Stage 2: chunk a normalized document into overlapping, boundary-aware pieces."""
from __future__ import annotations

from chonkie import OverlapRefinery, RecursiveChunker, RecursiveLevel, RecursiveRules
from pydantic import BaseModel

from highlighter.normalize import NormalizedDoc

_TOKENIZER = "cl100k_base"

# Markdown-aware splitting hierarchy. Each level is a fallback tried when the
# previous level doesn't yield small-enough chunks.
_MARKDOWN_RULES = RecursiveRules(
    levels=[
        # Headings: split before any ATX heading so each chunk starts at a heading.
        RecursiveLevel(
            delimiters=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### "],
            include_delim="next",
        ),
        # Paragraph breaks.
        RecursiveLevel(delimiters="\n\n", include_delim="next"),
        # Sentence ends.
        RecursiveLevel(delimiters=[". ", "? ", "! "], include_delim="prev"),
        # Last resort: any whitespace (words).
        RecursiveLevel(whitespace=True),
    ]
)


class Chunk(BaseModel):
    text: str
    line_start: int
    line_end: int
    section_path: list[str] = []


def _line_number_at(text: str, char_index: int) -> int:
    """Return 1-indexed line number of `char_index` in `text`."""
    return text.count("\n", 0, char_index) + 1


def chunk_document(
    doc: NormalizedDoc,
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    chunker = RecursiveChunker(
        tokenizer=_TOKENIZER,
        chunk_size=chunk_size,
        rules=_MARKDOWN_RULES,
    )
    refinery = OverlapRefinery(
        tokenizer=_TOKENIZER,
        context_size=chunk_overlap,
        mode="token",
        method="prefix",
        merge=True,
    )
    raw_chunks = refinery(chunker(doc.text))
    chunks: list[Chunk] = []
    for c in raw_chunks:
        # OverlapRefinery(merge=True) prepends previous-chunk tail into
        # c.text but leaves c.start_index pointing at the post-prefix split,
        # so subtract the prefix length to get the true start of c.text.
        effective_start = c.start_index - len(c.context or "")
        line_start = _line_number_at(doc.text, effective_start)
        line_end = _line_number_at(doc.text, max(c.end_index - 1, effective_start))
        chunks.append(Chunk(
            text=c.text,
            line_start=line_start,
            line_end=line_end,
            section_path=doc.section_path_for_line(line_start),
        ))
    return chunks
