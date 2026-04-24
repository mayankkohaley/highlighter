from pathlib import Path

import pytest

from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from tests.llm_helpers import canned_function_model, varying_function_model


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
        "document: doc.md\n"
        "cases:\n"
        "  - name: p\n"
        "    chunk_selector:\n"
        '      section_path: ["Top", "P"]\n'
        "    query:\n"
        "      question: What is required?\n"
        "    expected_excerpts:\n"
        '      - "Node.js 20 or later"\n'
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


def test_cli_report_includes_predicted_excerpt_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # An extra, non-matching prediction so precision < 1.0 — confirms we can
    # still see the offending text in the report to diagnose it.
    docs = tmp_path / "docs"
    cases = tmp_path / "cases"
    docs.mkdir()
    cases.mkdir()
    (docs / "doc.md").write_text(
        "# Top\n\n## P\n\nNode.js 20 or later is required. The sky is blue.\n"
    )
    (cases / "p.yaml").write_text(
        "document: doc.md\n"
        "cases:\n"
        "  - name: p\n"
        "    chunk_selector:\n"
        '      section_path: ["Top", "P"]\n'
        "    query:\n"
        "      question: What is required?\n"
        "    expected_excerpts:\n"
        '      - "Node.js 20 or later"\n'
    )

    from evals.__main__ import _main

    agent = build_extractor_agent()
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="Node.js 20 or later"),
            RawExcerpt(text="The sky is blue"),  # unmatched — precision miss
        ]
    )

    with agent.override(model=canned_function_model(extraction)):
        _main(
            ["prog", "--cases-dir", str(cases), "--docs-dir", str(docs)],
            extract_agent=agent,
        )

    out = capsys.readouterr().out
    assert "Node.js 20 or later" in out
    assert "The sky is blue" in out


def test_cli_runs_flag_reruns_each_case_and_reports_f1_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    cases = tmp_path / "cases"
    docs.mkdir()
    cases.mkdir()
    (docs / "doc.md").write_text("# Top\n\n## P\n\nNode.js 20 or later is required.\n")
    (cases / "p.yaml").write_text(
        "document: doc.md\n"
        "cases:\n"
        "  - name: p\n"
        "    chunk_selector:\n"
        '      section_path: ["Top", "P"]\n'
        "    query:\n"
        "      question: What is required?\n"
        "    expected_excerpts:\n"
        '      - "Node.js 20 or later"\n'
    )

    from evals.__main__ import _main

    agent = build_extractor_agent()
    # Alternate between a matching extract (F1=1) and nothing (F1=0).
    # Across 3 runs we'll see F1s of [1.0, 0.0, 1.0].
    outputs = [
        _ExtractorOutput(excerpts=[RawExcerpt(text="Node.js 20 or later")]),
        _ExtractorOutput(excerpts=[]),
        _ExtractorOutput(excerpts=[RawExcerpt(text="Node.js 20 or later")]),
    ]

    with agent.override(model=varying_function_model(outputs)):
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
                "--runs", "3",
            ],
            extract_agent=agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    # Proof we actually ran 3 times: min F1 came from the empty run, max from the matching ones.
    assert "min=0.00" in out
    assert "max=1.00" in out
    assert "3 runs" in out
