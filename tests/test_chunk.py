from pathlib import Path

from highlighter.chunk import chunk_document
from highlighter.normalize import normalize


def test_small_document_produces_single_chunk_with_full_text(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nA short body that fits in one chunk.\n")
    doc = normalize(md)

    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text.strip() == doc.text.strip()


def test_long_document_is_split_into_multiple_chunks(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    body = "\n\n".join(
        f"Paragraph {i} contains some filler text so it has a few tokens."
        for i in range(20)
    )
    md.write_text(f"# Title\n\n{body}\n")
    doc = normalize(md)

    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1


def test_chunks_carry_line_start_and_line_end(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    body = "\n\n".join(
        f"Paragraph {i} contains some filler text so it has a few tokens."
        for i in range(20)
    )
    md.write_text(f"# Title\n\n{body}\n")
    doc = normalize(md)

    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=10)

    assert len(chunks) >= 2
    assert chunks[0].line_start == 1
    assert chunks[1].line_start > chunks[0].line_start
    for c in chunks:
        assert 1 <= c.line_start <= c.line_end


def test_chunk_section_path_matches_doc_at_line_start(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    paragraphs = "\n\n".join(f"Filler paragraph {i}." for i in range(10))
    md.write_text(
        "# Top\n\n"
        "## Sub A\n\n"
        f"{paragraphs}\n\n"
        "## Sub B\n\n"
        f"{paragraphs}\n"
    )
    doc = normalize(md)

    chunks = chunk_document(doc, chunk_size=60, chunk_overlap=10)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.section_path == doc.section_path_for_line(c.line_start)
