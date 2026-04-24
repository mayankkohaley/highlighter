from pathlib import Path

from evals.pipeline.fixtures import PipelineCase
from evals.pipeline.runner import run_pipeline_case
from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import RawExcerpt, _ExtractorOutput, build_extractor_agent
from tests.llm_helpers import canned_function_model


def test_run_pipeline_case_scores_end_to_end(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc.md").write_text(
        "# Title\n\nNode.js 20 or later is required. The sky is blue.\n"
    )

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()

    expansion = _QueryExpansion(
        sub_questions=["What runtime is needed?"],
        rubric="Useful excerpts name specific software versions.",
    )
    extraction = _ExtractorOutput(
        excerpts=[RawExcerpt(text="Node.js 20 or later")]
    )

    case = PipelineCase(
        name="c1",
        document="doc.md",
        question="What do I need?",
        expected_excerpts=["Node.js 20 or later"],
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline_case(
            case,
            docs_dir=docs,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert result.case is case
    assert result.predicted == ["Node.js 20 or later"]
    assert result.score.f1 == 1.0
    # The expanded query is available so callers can inspect generated
    # sub-questions and rubric (that's the lever for Stage 1 iteration).
    assert result.query.question == "What do I need?"
    assert result.query.sub_questions == ["What runtime is needed?"]
    assert result.query.rubric.startswith("Useful excerpts")
