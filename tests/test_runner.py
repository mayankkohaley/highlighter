from pathlib import Path

from evals.fixtures import load_cases
from evals.runner import run_case
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from tests.llm_helpers import canned_function_model


def test_tracer_runs_one_case_and_scores_it(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text(
        "# Top\n\n"
        "## Prereqs\n\n"
        "Need Node.js 20 or later to install.\n"
    )
    (tmp_path / "case.yaml").write_text(
        "document: doc.md\n"
        "cases:\n"
        "  - name: prereqs-test\n"
        "    chunk_selector:\n"
        '      section_path: ["Top", "Prereqs"]\n'
        "    query:\n"
        "      question: What do I need?\n"
        "    expected_excerpts:\n"
        '      - "Node.js 20 or later"\n'
    )
    case = load_cases(tmp_path / "case.yaml")[0]

    agent = build_extractor_agent()
    extraction = _ExtractorOutput(
        excerpts=[RawExcerpt(text="Node.js 20 or later")]
    )

    with agent.override(model=canned_function_model(extraction)):
        result = run_case(case, docs_dir=tmp_path, extract_agent=agent)

    assert result.predicted == ["Node.js 20 or later"]
    assert result.score.precision == 1.0
    assert result.score.recall == 1.0
    assert result.missing_expected == []
