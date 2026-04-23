import hashlib
from pathlib import Path

from highlighter.normalize import normalize


def test_normalize_returns_file_contents(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("hello world\n")

    doc = normalize(md)

    assert doc.text == "hello world\n"


def test_normalize_converts_crlf_and_cr_to_lf(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_bytes(b"a\r\nb\rc\n")

    doc = normalize(md)

    assert doc.text == "a\nb\nc\n"


def test_normalize_strips_trailing_whitespace_per_line(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("alpha   \nbeta\t\ngamma\n")

    doc = normalize(md)

    assert doc.text == "alpha\nbeta\ngamma\n"


def test_content_hash_is_sha256_of_raw_bytes(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    raw = b"alpha   \r\nbeta\n"
    md.write_bytes(raw)

    doc = normalize(md)

    assert doc.content_hash == hashlib.sha256(raw).hexdigest()


def test_source_path_is_recorded(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("hi\n")

    doc = normalize(md)

    assert doc.source_path == str(md)


def test_single_atx_heading_becomes_a_section(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Hello\n\nbody\n")

    doc = normalize(md)

    assert len(doc.sections) == 1
    section = doc.sections[0]
    assert section.level == 1
    assert section.title == "Hello"
    assert section.line_start == 1


def test_multiple_atx_headings_at_different_levels(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Top\n\n## Middle\n\n### Leaf\n")

    doc = normalize(md)

    assert [(s.level, s.title, s.line_start) for s in doc.sections] == [
        (1, "Top", 1),
        (2, "Middle", 3),
        (3, "Leaf", 5),
    ]


def test_seven_or_more_hashes_is_not_a_heading(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("####### Not a heading\n")

    doc = normalize(md)

    assert doc.sections == []


def test_headings_inside_fenced_code_blocks_are_ignored(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(
        "# Real\n"
        "\n"
        "```\n"
        "# Not a heading\n"
        "## Also not\n"
        "```\n"
        "\n"
        "## After\n"
    )

    doc = normalize(md)

    assert [(s.level, s.title) for s in doc.sections] == [(1, "Real"), (2, "After")]
