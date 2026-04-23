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
