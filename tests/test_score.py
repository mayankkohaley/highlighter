from evals.score import score_case


def test_tracer_exact_match_scores_one() -> None:
    result = score_case(predicted=["Node.js 20 or later"], expected=["Node.js 20 or later"])

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_expected_phrase_as_substring_of_predicted_matches() -> None:
    result = score_case(
        predicted=["Node.js 20 or later. The AgentCore CLI is distributed as an npm package."],
        expected=["Node.js 20 or later"],
    )

    assert result.precision == 1.0
    assert result.recall == 1.0


def test_unrelated_prediction_drops_precision() -> None:
    # One prediction matches the expected phrase, one is unrelated junk.
    result = score_case(
        predicted=["Node.js 20 or later", "The sky is blue"],
        expected=["Node.js 20 or later"],
    )

    assert result.precision == 0.5
    assert result.recall == 1.0
