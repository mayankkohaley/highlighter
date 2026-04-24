"""Precision / recall / F1 scoring for excerpt-extraction eval cases."""
from __future__ import annotations

from pydantic import BaseModel


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
        1 for p in predicted if any(e in p for e in expected)
    )
    matched_expected = sum(
        1 for e in expected if any(e in p for p in predicted)
    )
    precision = matched_predicted / len(predicted) if predicted else 0.0
    recall = matched_expected / len(expected) if expected else 0.0
    return CaseScore(precision=precision, recall=recall, f1=_f1(precision, recall))
