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
