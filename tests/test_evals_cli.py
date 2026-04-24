from pathlib import Path

import pytest

from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from tests.llm_helpers import canned_function_model


def test_cli_runs_one_case_and_prints_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    cases = tmp_path / "cases"
    docs.mkdir()
    cases.mkdir()
    (docs / "doc.md").write_text("# Top\n\n## P\n\nNode.js 20 or later is required.\n")
    (cases / "p.yaml").write_text(
        "name: p\n"
        "document: doc.md\n"
        "chunk_selector:\n"
        '  section_path: ["Top", "P"]\n'
        "query:\n"
        "  question: What is required?\n"
        "expected_excerpts:\n"
        '  - "Node.js 20 or later"\n'
    )

    from evals.__main__ import _main

    agent = build_extractor_agent()
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="Node.js 20 or later")])

    with agent.override(model=canned_function_model(extraction)):
        rc = _main(
            ["prog", "--cases-dir", str(cases), "--docs-dir", str(docs)],
            extract_agent=agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "p" in out           # case name appears
    assert "precision" in out   # metric label appears
    assert "1.00" in out        # perfect score shown
    assert "Aggregate" in out   # summary line appears
