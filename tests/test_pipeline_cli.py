from pathlib import Path

import pytest

from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import RawExcerpt, _ExtractorOutput, build_extractor_agent
from tests.llm_helpers import canned_function_model, varying_function_model


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


def test_cli_runs_flag_reruns_each_case_and_reports_f1_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs, cases = _tiny_suite(tmp_path)

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    # Alternate between matching extract (F1=1) and empty (F1=0) across 3 runs.
    extractions = [
        _ExtractorOutput(excerpts=[RawExcerpt(text="Node.js 20 or later")]),
        _ExtractorOutput(excerpts=[]),
        _ExtractorOutput(excerpts=[RawExcerpt(text="Node.js 20 or later")]),
    ]

    from evals.pipeline.__main__ import _main

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=varying_function_model(extractions)),
    ):
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
                "--runs", "3",
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    assert "min=0.00" in out
    assert "max=1.00" in out
    assert "3 runs" in out


def test_cli_write_baseline_writes_json_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],  # noqa: ARG001
) -> None:
    docs, cases = _tiny_suite(tmp_path)
    baseline_path = tmp_path / "baseline-pipeline.json"
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(
        excerpts=[RawExcerpt(text="Node.js 20 or later")]
    )
    expand_agent, extract_agent, (expand_ov, extract_ov) = _overrides(expansion, extraction)

    from evals.baseline import load as load_baseline
    from evals.pipeline.__main__ import _main

    with expand_ov, extract_ov:
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
                "--write-baseline",
                "--baseline-path", str(baseline_path),
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert rc == 0
    assert baseline_path.exists()
    baseline = load_baseline(baseline_path)
    assert baseline.cases["runtime"].f1 == 1.0


def test_cli_check_baseline_exits_nonzero_on_regression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs, cases = _tiny_suite(tmp_path)
    baseline_path = tmp_path / "baseline-pipeline.json"
    baseline_path.write_text(
        '{"cases": {"runtime": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}}'
    )
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    # Empty extraction → current F1=0 vs baseline 1.0 → regression.
    extraction = _ExtractorOutput(excerpts=[])
    expand_agent, extract_agent, (expand_ov, extract_ov) = _overrides(expansion, extraction)

    from evals.pipeline.__main__ import _main

    with expand_ov, extract_ov:
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
                "--check-baseline",
                "--baseline-path", str(baseline_path),
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    out = capsys.readouterr().out
    assert rc != 0
    assert "regression" in out.lower()
    assert "runtime" in out


def test_cli_check_baseline_passes_when_scores_hold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],  # noqa: ARG001
) -> None:
    docs, cases = _tiny_suite(tmp_path)
    baseline_path = tmp_path / "baseline-pipeline.json"
    baseline_path.write_text(
        '{"cases": {"runtime": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}}'
    )
    expansion = _QueryExpansion(sub_questions=[], rubric="")
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
                "--check-baseline",
                "--baseline-path", str(baseline_path),
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert rc == 0


def test_cli_chunk_size_flag_splits_doc_into_multiple_chunks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    cases = tmp_path / "pipeline"
    docs.mkdir()
    cases.mkdir()
    paragraphs = "\n\n".join(f"anchor {i} body words." for i in range(30))
    (docs / "doc.md").write_text(f"# Title\n\n{paragraphs}\n")
    (cases / "case.yaml").write_text(
        "document: doc.md\n"
        "cases:\n"
        "  - name: c\n"
        '    question: "q"\n'
        "    expected_excerpts: []\n"
    )

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    # Canned extractor returns one "anchor" span per call; predicted count
    # equals chunks processed, so we can observe the flag's effect.
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="anchor")])

    from evals.pipeline.__main__ import _main

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        rc = _main(
            [
                "prog",
                "--cases-dir", str(cases),
                "--docs-dir", str(docs),
                "--chunk-size", "60",
                "--chunk-overlap", "10",
            ],
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    out = capsys.readouterr().out
    assert rc == 0
    anchor_lines = [line for line in out.splitlines() if line.strip() == "> anchor"]
    assert len(anchor_lines) > 1
