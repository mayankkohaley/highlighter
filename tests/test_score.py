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


def test_missing_expected_phrase_drops_recall() -> None:
    result = score_case(
        predicted=["Node.js 20 or later"],
        expected=["Node.js 20 or later", "Python 3.10 or later"],
    )

    assert result.precision == 1.0
    assert result.recall == 0.5


def test_empty_predicted_and_empty_expected_is_perfect() -> None:
    # "Nothing should be extracted from this chunk" and nothing was — vacuously correct.
    result = score_case(predicted=[], expected=[])

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_score_tolerates_markdown_emphasis_markers() -> None:
    # A predicted span that embeds emphasis markers inside what would otherwise
    # be the expected phrase must still be credited as a match.
    result = score_case(
        predicted=["the *Tsesarevich* were put out of action for weeks"],
        expected=["Tsesarevich were put out of action"],
    )

    assert result.precision == 1.0
    assert result.recall == 1.0
