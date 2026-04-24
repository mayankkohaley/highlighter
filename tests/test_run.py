from pathlib import Path

from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from highlighter.run import run_pipeline
from highlighter.synthesize import Synthesis, build_synthesis_agent
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


def test_pipeline_consolidates_adjacent_verified_excerpts(tmp_path: Path) -> None:
    # Stage 4 runs inside the pipeline: two adjacent excerpts in the same
    # section collapse to one consolidated span. The raw verified list is
    # preserved for downstream inspection.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nalpha line\nbeta line\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="alpha line"),
            RawExcerpt(text="beta line"),
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

    assert len(result.excerpts) == 2
    assert len(result.consolidated) == 1
    assert result.consolidated[0].text == "alpha line\nbeta line"
    assert result.consolidated[0].line_start == 3
    assert result.consolidated[0].line_end == 4


def test_pipeline_skips_synthesis_by_default(tmp_path: Path) -> None:
    # Synthesis costs tokens — default is opt-in.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nalpha line.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="alpha line.")])

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

    assert result.synthesis is None


def test_pipeline_runs_synthesis_when_opted_in(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nalpha line.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    synthesis_agent = build_synthesis_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="alpha line.")])
    canned_answer = Synthesis(answer="It is alpha. [1]", used_excerpts=[1])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
        synthesis_agent.override(model=canned_function_model(canned_answer)),
    ):
        result = run_pipeline(
            md,
            question="?",
            synthesize=True,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
            synthesis_agent=synthesis_agent,
        )

    assert result.synthesis is not None
    assert result.synthesis.answer == "It is alpha. [1]"
    assert result.synthesis.used_excerpts == [1]
