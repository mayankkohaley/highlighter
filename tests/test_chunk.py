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
