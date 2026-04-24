"""Substring matching that tolerates markdown emphasis markers.

Used by Stage 3 verification (was the span in the chunk?) and by the eval
scorer (did the predicted span cover this expected phrase?) so that neither
side is brittle to `*bold*`, `_italic_`, or `` `code` `` markers appearing
in one string but not the other.

Markers are stripped symmetrically from both sides before the substring check.
"""
from __future__ import annotations

import re

_EMPHASIS_MARKERS = re.compile(r"[*_`]")


def _strip(text: str) -> str:
    return _EMPHASIS_MARKERS.sub("", text)


def find_span(haystack: str, needle: str) -> tuple[int, int] | None:
    r"""Locate `needle` inside `haystack` after stripping `* _ \``.

    Returns `(start, end)` character offsets into the ORIGINAL `haystack`
    (so callers can map back to lines), or None if not found.
    """
    stripped_needle = _strip(needle)
    if not stripped_needle:
        return None
    mapping: list[int] = []
    kept: list[str] = []
    for i, ch in enumerate(haystack):
        if ch in "*_`":
            continue
        kept.append(ch)
        mapping.append(i)
    stripped_haystack = "".join(kept)
    idx = stripped_haystack.find(stripped_needle)
    if idx == -1:
        return None
    start = mapping[idx]
    end = mapping[idx + len(stripped_needle) - 1] + 1
    return start, end


def contains(haystack: str, needle: str) -> bool:
    r"""True if `needle` appears in `haystack` after stripping `* _ \``."""
    return find_span(haystack, needle) is not None
