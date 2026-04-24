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


def test_run_pipeline_case_respects_custom_chunk_size(tmp_path: Path) -> None:
    # A multi-paragraph doc that fits in one default chunk (2000 tokens) but
    # splits into multiple chunks at a very small chunk_size. A canned extractor
    # returning one "anchor" span per call gives us a way to observe chunk
    # count: len(predicted) == number of chunks the extractor saw.
    docs = tmp_path / "docs"
    docs.mkdir()
    paragraphs = "\n\n".join(f"anchor {i} body words." for i in range(30))
    (docs / "doc.md").write_text(f"# Title\n\n{paragraphs}\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    # Every chunk contains "anchor" — canned extractor returns it each call.
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="anchor")])

    case = PipelineCase(
        name="c",
        document="doc.md",
        question="?",
        expected_excerpts=[],
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        small = run_pipeline_case(
            case,
            docs_dir=docs,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
            chunk_size=60,
            chunk_overlap=10,
        )
        large = run_pipeline_case(
            case,
            docs_dir=docs,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
            chunk_size=5000,
            chunk_overlap=10,
        )

    assert len(small.predicted) > len(large.predicted)
    assert len(large.predicted) == 1
