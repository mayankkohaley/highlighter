from pathlib import Path

from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from highlighter.run import run_pipeline
from tests.llm_helpers import canned_function_model


def test_tracer_runs_full_pipeline(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(
        sub_questions=["What animal is described?"],
        rubric="Useful excerpts name an animal.",
    )
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(
                text="quick brown fox",
                which_subquestion="What animal is described?",
                confidence=0.9,
            )
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="What animal is described?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert result.query.question == "What animal is described?"
    assert result.query.sub_questions == ["What animal is described?"]
    assert len(result.excerpts) == 1
    assert result.excerpts[0].text == "quick brown fox"
