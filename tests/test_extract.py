from pathlib import Path

from highlighter.chunk import chunk_document
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
    extract_excerpts,
)
from highlighter.normalize import normalize
from highlighter.query import Query
from tests.llm_helpers import canned_function_model


def _canned(candidates: list[RawExcerpt]):
    return canned_function_model(_ExtractorOutput(excerpts=candidates))


def test_extractor_agent_pins_temperature_to_zero_for_determinism() -> None:
    agent = build_extractor_agent()

    assert agent.model_settings.get("temperature") == 0.0


def test_verification_tolerates_stripped_markdown_emphasis(tmp_path: Path) -> None:
    # The chunk wraps "Python 3.10 or later" in bold asterisks.
    md = tmp_path / "doc.md"
    md.write_text(
        "# X\n\n"
        "+  **Python 3.10 or later** (for agent code).\n"
    )
    doc = normalize(md)
    chunks = chunk_document(doc)
    chunk = chunks[0]

    agent = build_extractor_agent()
    # Model returned the same span without the asterisks — should still verify.
    candidates = [RawExcerpt(text="Python 3.10 or later (for agent code)")]

    with agent.override(model=_canned(candidates)):
        excerpts = extract_excerpts(chunk, Query(question="?"), doc, agent=agent)

    assert len(excerpts) == 1
    assert excerpts[0].text == "Python 3.10 or later (for agent code)"


def test_tracer_returns_one_verified_excerpt(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")
    doc = normalize(md)
    chunks = chunk_document(doc)
    chunk = chunks[0]

    agent = build_extractor_agent()
    query = Query(question="What animal jumps?")
    candidates = [RawExcerpt(text="The quick brown fox jumps over the lazy dog.")]

    with agent.override(model=_canned(candidates)):
        excerpts = extract_excerpts(chunk, query, doc, agent=agent)

    assert len(excerpts) == 1
    assert excerpts[0].text == "The quick brown fox jumps over the lazy dog."


def test_empty_candidates_return_empty_list(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nIrrelevant body.\n")
    doc = normalize(md)
    chunk = chunk_document(doc)[0]

    agent = build_extractor_agent()
    with agent.override(model=_canned([])):
        excerpts = extract_excerpts(chunk, Query(question="anything"), doc, agent=agent)

    assert excerpts == []


def test_excerpt_citation_reflects_its_span_inside_the_chunk(tmp_path: Path) -> None:
    # The chunk spans two sibling sections. A candidate pulled from the SECOND
    # section must be cited as belonging to that section — not to the chunk's
    # starting section (which is what a naive inherit-from-chunk would give).
    md = tmp_path / "doc.md"
    md.write_text(
        "# Top\n"
        "\n"
        "## First\n"
        "\n"
        "alpha line.\n"
        "\n"
        "## Second\n"
        "\n"
        "beta line here.\n"
    )
    doc = normalize(md)
    chunks = chunk_document(doc)
    assert len(chunks) == 1  # tracer requires a single chunk covering both sections
    chunk = chunks[0]
    assert chunk.section_path == ["Top"]

    agent = build_extractor_agent()
    candidates = [RawExcerpt(text="beta line here.")]
    with agent.override(model=_canned(candidates)):
        excerpts = extract_excerpts(chunk, Query(question="?"), doc, agent=agent)

    assert len(excerpts) == 1
    e = excerpts[0]
    assert e.section_path == ["Top", "Second"]
    assert e.line_start == 9
    assert e.line_end == 9


def test_which_subquestion_and_confidence_propagate(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")
    doc = normalize(md)
    chunk = chunk_document(doc)[0]

    agent = build_extractor_agent()
    candidates = [
        RawExcerpt(
            text="quick brown fox",
            which_subquestion="Which animal is described?",
            confidence=0.82,
        )
    ]
    with agent.override(model=_canned(candidates)):
        excerpts = extract_excerpts(chunk, Query(question="?"), doc, agent=agent)

    assert len(excerpts) == 1
    assert excerpts[0].which_subquestion == "Which animal is described?"
    assert excerpts[0].confidence == 0.82


def test_non_substring_candidates_are_dropped(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")
    doc = normalize(md)
    chunk = chunk_document(doc)[0]

    agent = build_extractor_agent()
    candidates = [
        RawExcerpt(text="The quick brown fox"),                     # substring → keep
        RawExcerpt(text="An eagle soars above the mountain peak"),  # hallucination → drop
    ]

    with agent.override(model=_canned(candidates)):
        excerpts = extract_excerpts(chunk, Query(question="anything"), doc, agent=agent)

    assert [e.text for e in excerpts] == ["The quick brown fox"]
