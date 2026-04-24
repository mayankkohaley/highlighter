"""Stage 4: consolidate verified excerpts into a non-overlapping, merged set.

Same-section excerpts whose line ranges overlap — or sit within `gap` lines of
each other — fuse into a single excerpt. The fused text is re-sliced from
`doc.text` (never stitched from the pieces) so we preserve verbatim source.
"""
from __future__ import annotations

from highlighter.extract import Excerpt
from highlighter.normalize import NormalizedDoc


def _slice_lines(text: str, line_start: int, line_end: int) -> str:
    lines = text.split("\n")
    return "\n".join(lines[line_start - 1:line_end])


def _merge(a: Excerpt, b: Excerpt, doc: NormalizedDoc) -> Excerpt:
    line_start = min(a.line_start, b.line_start)
    line_end = max(a.line_end, b.line_end)
    return Excerpt(
        text=_slice_lines(doc.text, line_start, line_end),
        which_subquestion=a.which_subquestion,
        confidence=max(a.confidence, b.confidence),
        line_start=line_start,
        line_end=line_end,
        section_path=a.section_path,
    )


def consolidate(
    excerpts: list[Excerpt],
    doc: NormalizedDoc,
    *,
    gap: int = 2,
) -> list[Excerpt]:
    ordered = sorted(excerpts, key=lambda e: (e.line_start, e.line_end))
    merged: list[Excerpt] = []
    for e in ordered:
        if merged:
            prev = merged[-1]
            if (
                prev.section_path == e.section_path
                and prev.line_end + gap >= e.line_start
            ):
                merged[-1] = _merge(prev, e, doc)
                continue
        merged.append(e)
    return merged
