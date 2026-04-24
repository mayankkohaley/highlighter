"""Baseline tracking and regression gate for eval cases."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator

from evals.runner import CaseResult

_SCORE_DECIMALS = 3


class CaseBaseline(BaseModel):
    precision: float
    recall: float
    f1: float

    @field_validator("precision", "recall", "f1")
    @classmethod
    def _round(cls, v: float) -> float:
        return round(v, _SCORE_DECIMALS)


class Baseline(BaseModel):
    cases: dict[str, CaseBaseline]


class Regression(BaseModel):
    case_name: str
    baseline_f1: float
    current_f1: float

    @property
    def delta(self) -> float:
        return self.current_f1 - self.baseline_f1


DEFAULT_TOLERANCE = 0.02


def save(baseline: Baseline, path: Path | str) -> None:
    Path(path).write_text(baseline.model_dump_json(indent=2) + "\n")


def load(path: Path | str) -> Baseline:
    return Baseline.model_validate_json(Path(path).read_text())


def aggregate(runs_per_case: list[list[CaseResult]]) -> Baseline:
    """Compute mean precision/recall/F1 per case across its runs.

    All runs in a sub-list share a case name; the first run supplies it.
    """
    cases: dict[str, CaseBaseline] = {}
    for runs in runs_per_case:
        if not runs:
            continue
        name = runs[0].case.name
        n = len(runs)
        cases[name] = CaseBaseline(
            precision=sum(r.score.precision for r in runs) / n,
            recall=sum(r.score.recall for r in runs) / n,
            f1=sum(r.score.f1 for r in runs) / n,
        )
    return Baseline(cases=cases)


def check_regressions(
    baseline: Baseline,
    current: Baseline,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Regression]:
    regressions: list[Regression] = []
    for name, base in baseline.cases.items():
        cur = current.cases.get(name)
        current_f1 = cur.f1 if cur is not None else 0.0
        if current_f1 < base.f1 - tolerance:
            regressions.append(Regression(
                case_name=name,
                baseline_f1=base.f1,
                current_f1=current_f1,
            ))
    return regressions
