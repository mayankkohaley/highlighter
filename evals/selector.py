"""Pick a Chunk out of a NormalizedDoc for a given ChunkSelector."""
from __future__ import annotations

from evals.fixtures import ChunkSelector
from highlighter.chunk import Chunk
from highlighter.normalize import NormalizedDoc


def select_chunk(doc: NormalizedDoc, selector: ChunkSelector) -> Chunk:
    """Return a Chunk spanning the section whose heading path matches.

    The chunk is built directly from the section's line range, not via
    `chunk_document`. Evals stay deterministic under chunker-config changes.
    """
    lines = doc.text.split("\n")
    for section in doc.sections:
        path = doc.section_path_for_line(section.line_start)
        if path == selector.section_path:
            text = "\n".join(lines[section.line_start - 1 : section.line_end])
            return Chunk(
                text=text,
                line_start=section.line_start,
                line_end=section.line_end,
                section_path=path,
            )
    raise ValueError(f"no section matches section_path={selector.section_path}")
