from highlighter.matching import contains, find_span


def test_contains_tolerates_markdown_link_syntax_in_haystack() -> None:
    # The Stage 3 extractor routinely simplifies `[text](url)` → `text` when
    # quoting verbatim. Substring verification must still recognize those
    # candidates as present in the original document.
    haystack = "Install from [nodejs.org](https://nodejs.org) if needed."
    needle = "Install from nodejs.org if needed."

    assert contains(haystack, needle)


def test_find_span_maps_stripped_link_match_back_to_original_bounds() -> None:
    haystack = "prefix [nodejs.org](https://nodejs.org) suffix"
    needle = "prefix nodejs.org suffix"

    span = find_span(haystack, needle)

    assert span is not None
    start, end = span
    # The reported span covers the entire original substring, including the
    # markdown link syntax — so downstream citation line math lands correctly.
    assert haystack[start:end] == haystack
