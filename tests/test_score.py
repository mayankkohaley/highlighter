from evals.score import score_case


def test_tracer_exact_match_scores_one() -> None:
    result = score_case(predicted=["Node.js 20 or later"], expected=["Node.js 20 or later"])

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
