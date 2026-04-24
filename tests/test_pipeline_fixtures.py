from pathlib import Path

from evals.pipeline.fixtures import load_pipeline_cases


def test_load_pipeline_cases_stamps_shared_document_onto_each_case(tmp_path: Path) -> None:
    yaml_path = tmp_path / "fixture.yaml"
    yaml_path.write_text(
        "document: some-doc.md\n"
        "cases:\n"
        "  - name: first\n"
        "    question: What is X?\n"
        "    expected_excerpts:\n"
        "      - foo\n"
        "      - bar\n"
        "  - name: second\n"
        "    question: Why Y?\n"
        "    expected_excerpts:\n"
        "      - baz\n"
    )

    cases = load_pipeline_cases(yaml_path)

    assert [c.name for c in cases] == ["first", "second"]
    assert all(c.document == "some-doc.md" for c in cases)
    assert cases[0].question == "What is X?"
    assert cases[0].expected_excerpts == ["foo", "bar"]
    assert cases[1].expected_excerpts == ["baz"]


def test_load_pipeline_cases_treats_all_comments_file_as_no_cases(tmp_path: Path) -> None:
    yaml_path = tmp_path / "fixture.yaml"
    yaml_path.write_text("# document: nothing.md\n# cases: []\n")

    assert load_pipeline_cases(yaml_path) == []
