from pathlib import Path

from evals.fixtures import load_case


def test_tracer_loads_minimal_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "case.yaml"
    yaml_path.write_text(
        "name: example\n"
        "document: agentcore-get-started-cli.md\n"
        "chunk_selector:\n"
        '  section_path: ["A", "B"]\n'
        "query:\n"
        "  question: What is X?\n"
        "expected_excerpts:\n"
        "  - foo\n"
    )

    case = load_case(yaml_path)

    assert case.name == "example"
    assert case.document == "agentcore-get-started-cli.md"
    assert case.chunk_selector.section_path == ["A", "B"]
    assert case.query.question == "What is X?"
    assert case.expected_excerpts == ["foo"]
