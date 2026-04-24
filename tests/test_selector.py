from pathlib import Path

from evals.fixtures import ChunkSelector
from evals.selector import select_chunk
from highlighter.normalize import normalize


def test_tracer_selects_chunk_whose_section_path_matches(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(
        "# Top\n\n"
        "## Section A\n\n"
        "Content of section A.\n\n"
        "## Section B\n\n"
        "Content of section B.\n"
    )
    doc = normalize(md)
    selector = ChunkSelector(section_path=["Top", "Section A"])

    chunk = select_chunk(doc, selector)

    assert chunk.section_path == ["Top", "Section A"]
    assert "Content of section A" in chunk.text
    assert "Content of section B" not in chunk.text
