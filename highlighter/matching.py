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


def contains(haystack: str, needle: str) -> bool:
    """True if `needle` appears in `haystack` after stripping `* _ \``."""
    return _EMPHASIS_MARKERS.sub("", needle) in _EMPHASIS_MARKERS.sub("", haystack)
