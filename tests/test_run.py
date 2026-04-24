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


def test_excerpts_aggregate_across_all_chunks(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    body = "\n\n".join(f"Filler paragraph {i}." for i in range(40))
    md.write_text(
        "# Title\n\n"
        "MARKER_ALPHA appears early.\n\n"
        f"{body}\n\n"
        "MARKER_BETA appears late.\n"
    )

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="MARKER_ALPHA appears early."),
            RawExcerpt(text="MARKER_BETA appears late."),
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="?",
            chunk_size=80,
            chunk_overlap=10,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    texts = {e.text for e in result.excerpts}
    assert "MARKER_ALPHA appears early." in texts
    assert "MARKER_BETA appears late." in texts


def test_pipeline_result_carries_raw_candidates_including_dropped(tmp_path: Path) -> None:
    # Diagnosing the extractor requires seeing every span the model returned,
    # including ones that failed substring verification. The pipeline result
    # must surface those pre-verification candidates.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    # One span that will verify; one that won't.
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="quick brown fox"),
            RawExcerpt(text="HALLUCINATED phrase not in doc"),
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    raw_texts = [c.text for c in result.raw_candidates]
    assert "quick brown fox" in raw_texts
    assert "HALLUCINATED phrase not in doc" in raw_texts
    # Verification still drops the hallucination from excerpts.
    assert [e.text for e in result.excerpts] == ["quick brown fox"]
