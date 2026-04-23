from pathlib import Path

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from highlighter.chunk import chunk_document
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
    extract_excerpts,
)
from highlighter.normalize import normalize
from highlighter.query import Query


def _canned(candidates: list[RawExcerpt]) -> FunctionModel:
    output = _ExtractorOutput(excerpts=candidates)

    def fn(messages, info):
        return ModelResponse(parts=[TextPart(content=output.model_dump_json())])

    return FunctionModel(fn)


def test_tracer_returns_one_verified_excerpt(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")
    doc = normalize(md)
    chunks = chunk_document(doc)
    chunk = chunks[0]

    agent = build_extractor_agent()
    query = Query(question="What animal jumps?")
    candidates = [RawExcerpt(text="The quick brown fox jumps over the lazy dog.")]

    with agent.override(model=_canned(candidates)):
        excerpts = extract_excerpts(chunk, query, agent=agent)

    assert len(excerpts) == 1
    assert excerpts[0].text == "The quick brown fox jumps over the lazy dog."
