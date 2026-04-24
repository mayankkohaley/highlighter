from highlighter.consolidate import consolidate
from highlighter.extract import Excerpt
from highlighter.normalize import NormalizedDoc


def _doc(text: str) -> NormalizedDoc:
    return NormalizedDoc(source_path="t.md", content_hash="x", text=text, sections=[])


def _excerpt(
    text: str,
    *,
    line_start: int,
    line_end: int,
    section_path: list[str] | None = None,
    confidence: float = 1.0,
) -> Excerpt:
    return Excerpt(
        text=text,
        line_start=line_start,
        line_end=line_end,
        section_path=section_path or [],
        confidence=confidence,
    )


def test_consolidate_empty_list_returns_empty() -> None:
    assert consolidate([], _doc("")) == []


def test_consolidate_single_excerpt_passes_through_unchanged() -> None:
    e = _excerpt("hello", line_start=1, line_end=1)
    assert consolidate([e], _doc("hello")) == [e]


def test_consolidate_merges_overlapping_excerpts_in_same_section() -> None:
    # Overlapping line ranges in the same section → fuse into one excerpt whose
    # text is re-sliced from the doc so we never glue strings manually.
    doc = _doc("alpha\nbeta\ngamma\ndelta\n")
    a = _excerpt("alpha\nbeta", line_start=1, line_end=2, section_path=["S"],
                 confidence=0.7)
    b = _excerpt("beta\ngamma", line_start=2, line_end=3, section_path=["S"],
                 confidence=0.9)

    result = consolidate([a, b], doc)

    assert len(result) == 1
    merged = result[0]
    assert merged.line_start == 1
    assert merged.line_end == 3
    assert merged.text == "alpha\nbeta\ngamma"
    assert merged.section_path == ["S"]
    assert merged.confidence == 0.9


def test_consolidate_merges_across_small_gap_within_threshold() -> None:
    # Two excerpts separated by a blank line (gap=1) in the same section merge.
    # Default gap threshold is 2, so this case must fuse.
    doc = _doc("alpha\n\ngamma\n")
    a = _excerpt("alpha", line_start=1, line_end=1, section_path=["S"])
    b = _excerpt("gamma", line_start=3, line_end=3, section_path=["S"])

    result = consolidate([a, b], doc)

    assert len(result) == 1
    assert result[0].line_start == 1
    assert result[0].line_end == 3
    assert result[0].text == "alpha\n\ngamma"


def test_consolidate_keeps_excerpts_separated_by_gap_beyond_threshold() -> None:
    # Default gap=2; a 3-line gap exceeds it — both excerpts survive intact.
    doc = _doc("alpha\n\n\n\ndelta\n")
    a = _excerpt("alpha", line_start=1, line_end=1, section_path=["S"])
    b = _excerpt("delta", line_start=5, line_end=5, section_path=["S"])

    result = consolidate([a, b], doc)

    assert result == [a, b]


def test_consolidate_sorts_output_by_line_start() -> None:
    doc = _doc("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n")
    late = _excerpt("j", line_start=10, line_end=10, section_path=["S"])
    early = _excerpt("a", line_start=1, line_end=1, section_path=["S"])
    mid = _excerpt("e", line_start=5, line_end=5, section_path=["S"])

    result = consolidate([late, early, mid], doc)

    assert [e.line_start for e in result] == [1, 5, 10]


def test_consolidate_does_not_merge_across_section_boundary() -> None:
    # Adjacent lines but different section_path — a heading sits between them,
    # so merging would blur the citation. Keep them separate.
    doc = _doc("alpha\nbeta\n")
    a = _excerpt("alpha", line_start=1, line_end=1, section_path=["S1"])
    b = _excerpt("beta", line_start=2, line_end=2, section_path=["S2"])

    result = consolidate([a, b], doc)

    assert result == [a, b]
