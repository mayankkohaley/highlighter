"""Substring matching that tolerates markdown formatting.

Used by Stage 3 verification (was the span in the chunk?) and by the eval
scorer (did the predicted span cover this expected phrase?) so that neither
side is brittle to markdown surface syntax that the model may strip when
quoting verbatim:

- Emphasis markers (`*`, `_`, `` ` ``) — stripped.
- Markdown links `[text](url)` — compressed to just `text`.

Both transforms are applied symmetrically to haystack and needle. `find_span`
tracks original offsets so callers (e.g. per-excerpt citation) can map a
match back to its real position in the source text.
"""
from __future__ import annotations

import re

_EMPHASIS_MARKERS = frozenset("*_`")
_MD_LINK_RE = re.compile(r"\[(.+?)\]\(.+?\)")


def _compress(text: str) -> tuple[str, list[int]]:
    """Return the text with emphasis markers and markdown link syntax stripped,
    along with a mapping from compressed-index → original-index.

    Inside a `[text](url)` link, only the inner `text` is emitted, and each
    kept character is mapped back to its position inside the original
    brackets — so a match on the compressed string can be projected back to
    the full original span (brackets, url, and all).
    """
    kept: list[str] = []
    mapping: list[int] = []

    def _emit_range(start: int, end: int) -> None:
        for i in range(start, end):
            ch = text[i]
            if ch in _EMPHASIS_MARKERS:
                continue
            kept.append(ch)
            mapping.append(i)

    pos = 0
    for m in _MD_LINK_RE.finditer(text):
        _emit_range(pos, m.start())
        # Inner link text starts right after the opening `[`.
        inner_start = m.start() + 1
        for j, ch in enumerate(m.group(1)):
            if ch in _EMPHASIS_MARKERS:
                continue
            kept.append(ch)
            mapping.append(inner_start + j)
        pos = m.end()
    _emit_range(pos, len(text))

    return "".join(kept), mapping


def find_span(haystack: str, needle: str) -> tuple[int, int] | None:
    """Locate `needle` inside `haystack`, tolerating emphasis markers and
    markdown links on either side.

    Returns `(start, end)` character offsets into the ORIGINAL `haystack`
    (so callers can map back to lines), or None if not found.
    """
    stripped_needle, _ = _compress(needle)
    if not stripped_needle:
        return None
    stripped_haystack, mapping = _compress(haystack)
    idx = stripped_haystack.find(stripped_needle)
    if idx == -1:
        return None
    start = mapping[idx]
    end = mapping[idx + len(stripped_needle) - 1] + 1
    return start, end


def contains(haystack: str, needle: str) -> bool:
    """True if `needle` appears in `haystack` after emphasis/link stripping."""
    return find_span(haystack, needle) is not None
