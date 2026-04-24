"""Precision / recall / F1 scoring for excerpt-extraction eval cases."""
from __future__ import annotations

import re

from pydantic import BaseModel

# Same tolerance as extract.py verification — strip markdown emphasis markers
# symmetrically so `*Word*` in a predicted span doesn't break substring match
# against an expected phrase that crosses the emphasized word.
_EMPHASIS_MARKERS = re.compile(r"[*_`]")


def _contains(haystack: str, needle: str) -> bool:
    return _EMPHASIS_MARKERS.sub("", needle) in _EMPHASIS_MARKERS.sub("", haystack)


class CaseScore(BaseModel):
    precision: float
    recall: float
    f1: float


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_case(predicted: list[str], expected: list[str]) -> CaseScore:
    if not predicted and not expected:
        # Nothing should be extracted, nothing was — vacuously correct.
        return CaseScore(precision=1.0, recall=1.0, f1=1.0)
    matched_predicted = sum(
        1 for p in predicted if any(_contains(p, e) for e in expected)
    )
    matched_expected = sum(
        1 for e in expected if any(_contains(p, e) for p in predicted)
    )
    precision = matched_predicted / len(predicted) if predicted else 0.0
    recall = matched_expected / len(expected) if expected else 0.0
    return CaseScore(precision=precision, recall=recall, f1=_f1(precision, recall))
