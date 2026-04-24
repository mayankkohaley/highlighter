from pathlib import Path

from evals.fixtures import load_cases


def test_load_cases_returns_all_cases_with_document_filled_in(tmp_path: Path) -> None:
    yaml_path = tmp_path / "fixture.yaml"
    yaml_path.write_text(
        "document: agentcore-get-started-cli.md\n"
        "cases:\n"
        "  - name: first\n"
        "    chunk_selector:\n"
        '      section_path: ["A"]\n'
        "    query:\n"
        "      question: What?\n"
        "    expected_excerpts:\n"
        "      - foo\n"
        "  - name: second\n"
        "    chunk_selector:\n"
        '      section_path: ["B"]\n'
        "    query:\n"
        "      question: Why?\n"
        "    expected_excerpts:\n"
        "      - bar\n"
    )

    cases = load_cases(yaml_path)

    assert [c.name for c in cases] == ["first", "second"]
    assert all(c.document == "agentcore-get-started-cli.md" for c in cases)
    assert cases[0].chunk_selector.section_path == ["A"]
    assert cases[1].query.question == "Why?"
