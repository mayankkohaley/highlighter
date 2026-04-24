from pathlib import Path

import pytest

from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import RawExcerpt, _ExtractorOutput, build_extractor_agent
from tests.llm_helpers import canned_function_model


def _tiny_suite(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    cases = tmp_path / "pipeline"
    docs.mkdir()
    cases.mkdir()
    (docs / "doc.md").write_text(
        "# Title\n\nNode.js 20 or later is required. The sky is blue.\n"
    )
    (cases / "case.yaml").write_text(
        "document: doc.md\n"
        "cases:\n"
        "  - name: runtime\n"
        "    question: What is required?\n"
        "    expected_excerpts:\n"
        "      - Node.js 20 or later\n"
    )
    return docs, cases


def _overrides(expand_out: _QueryExpansion, extract_out: _ExtractorOutput):
    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    return expand_agent, extract_agent, (
        expand_agent.override(model=canned_function_model(expand_out)),
        extract_agent.override(model=canned_function_model(extract_out)),
    )


def test_cli_runs_one_pipeline_case_and_prints_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs, cases = _tiny_suite(tmp_path)
    expansion = _QueryExpansion(
        sub_questions=["What runtime is required?"],
        rubric="Useful excerpts name software versions.",
    )
    extraction = _ExtractorOutput(
        excerpts=[RawExcerpt(text="Node.js 20 or later")]
    )
    expand_agent, extract_agent, (expand_ov, extract_ov) = _overrides(expansion, extraction)

    from evals.pipeline.__main__ import _main

    with expand_ov, extract_ov:
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "runtime" in out
    assert "precision" in out
    assert "1.00" in out
    assert "Node.js 20 or later" in out
    assert "Aggregate" in out


def test_cli_debug_prints_generated_subquestions_and_rubric(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs, cases = _tiny_suite(tmp_path)
    expansion = _QueryExpansion(
        sub_questions=["SUB_Q_PROBE"],
        rubric="RUBRIC_PROBE",
    )
    extraction = _ExtractorOutput(
        excerpts=[RawExcerpt(text="Node.js 20 or later")]
    )
    expand_agent, extract_agent, (expand_ov, extract_ov) = _overrides(expansion, extraction)

    from evals.pipeline.__main__ import _main

    with expand_ov, extract_ov:
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
                "--debug",
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "SUB_Q_PROBE" in out
    assert "RUBRIC_PROBE" in out
