from pathlib import Path

from evals.baseline import (
    Baseline,
    CaseBaseline,
    aggregate,
    check_regressions,
    load,
    save,
)
from evals.fixtures import ChunkSelector, EvalCase
from evals.runner import CaseResult
from evals.score import CaseScore
from highlighter.query import Query


def _result(name: str, *, precision: float, recall: float, f1: float) -> CaseResult:
    return CaseResult(
        case=EvalCase(
            name=name,
            document="x.md",
            chunk_selector=ChunkSelector(section_path=["X"]),
            query=Query(question="?"),
            expected_excerpts=[],
        ),
        score=CaseScore(precision=precision, recall=recall, f1=f1),
        predicted=[],
        matched_expected=[],
        missing_expected=[],
    )


def _b(**cases: float) -> Baseline:
    return Baseline(cases={
        k: CaseBaseline(precision=v, recall=v, f1=v) for k, v in cases.items()
    })


def test_check_regressions_flags_case_whose_f1_dropped_beyond_tolerance() -> None:
    baseline = _b(alpha=0.90)
    current = _b(alpha=0.80)  # 0.10 drop, well beyond default tolerance

    regressions = check_regressions(baseline, current)

    assert len(regressions) == 1
    r = regressions[0]
    assert r.case_name == "alpha"
    assert r.baseline_f1 == 0.90
    assert r.current_f1 == 0.80


def test_check_regressions_tolerates_drop_within_tolerance() -> None:
    baseline = _b(alpha=0.90)
    current = _b(alpha=0.89)  # 0.01 drop, within default 0.02 tolerance

    assert check_regressions(baseline, current) == []


def test_check_regressions_flags_case_missing_from_current_run() -> None:
    # A case that was baselined but isn't in the current run (deleted fixture,
    # runtime error, typo in --case filter) should be treated as a regression —
    # silent-skip would hide real breakage.
    baseline = _b(alpha=0.90, beta=0.85)
    current = _b(alpha=0.90)  # beta is gone

    regressions = check_regressions(baseline, current)

    assert [r.case_name for r in regressions] == ["beta"]
    assert regressions[0].baseline_f1 == 0.85
    assert regressions[0].current_f1 == 0.0


def test_check_regressions_ignores_cases_only_in_current() -> None:
    # A new case added to the suite shouldn't break the gate before it's
    # baselined — only known-and-regressed cases should fail.
    baseline = _b(alpha=0.90)
    current = _b(alpha=0.90, beta=0.30)

    assert check_regressions(baseline, current) == []


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    baseline = Baseline(cases={
        "alpha": CaseBaseline(precision=0.95, recall=0.90, f1=0.925),
        "beta": CaseBaseline(precision=1.0, recall=0.80, f1=0.89),
    })

    save(baseline, path)
    loaded = load(path)

    assert loaded == baseline
    assert path.read_text().startswith("{")


def test_case_baseline_rounds_metrics_to_three_decimals() -> None:
    # Keeps the serialized baseline.json human-readable instead of e.g.
    # "precision": 0.6666666666666666. Applied at the model boundary so
    # every Baseline (aggregated or hand-loaded) holds the same precision.
    cb = CaseBaseline(precision=0.6666666666, recall=0.923076923, f1=0.7857142857)

    assert cb.precision == 0.667
    assert cb.recall == 0.923
    assert cb.f1 == 0.786


def test_aggregate_computes_mean_metrics_per_case_across_runs() -> None:
    runs_per_case = [
        [
            _result("alpha", precision=1.0, recall=1.0, f1=1.0),
            _result("alpha", precision=0.5, recall=0.5, f1=0.5),
        ],
        [
            _result("beta", precision=0.8, recall=0.6, f1=0.7),
        ],
    ]

    baseline = aggregate(runs_per_case)

    assert baseline.cases["alpha"].f1 == 0.75
    assert baseline.cases["alpha"].precision == 0.75
    assert baseline.cases["alpha"].recall == 0.75
    assert baseline.cases["beta"].f1 == 0.7
