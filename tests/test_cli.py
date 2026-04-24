from pathlib import Path

import pytest

from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from tests.llm_helpers import canned_function_model


def test_cli_prints_question_sub_questions_and_excerpts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")

    from highlighter.__main__ import _main

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(
        sub_questions=["Which animal is described?"],
        rubric="Useful excerpts name an animal.",
    )
    extraction = _ExtractorOutput(
        excerpts=[RawExcerpt(text="quick brown fox", confidence=0.9)]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        rc = _main(
            ["prog", str(md), "-q", "What animal is described?"],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "What animal is described?" in out
    assert "Which animal is described?" in out
    assert "quick brown fox" in out


def test_cli_missing_question_exits_with_error(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("irrelevant\n")

    from highlighter.__main__ import _main

    with pytest.raises(SystemExit) as exc:
        _main(["prog", str(md)])

    assert exc.value.code != 0
