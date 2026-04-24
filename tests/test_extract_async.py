import asyncio
from pathlib import Path

from highlighter.chunk import chunk_document
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
    extract_excerpts_verbose,
    extract_excerpts_verbose_async,
)
from highlighter.normalize import normalize
from highlighter.query import Query
from tests.llm_helpers import canned_function_model


def _canned(candidates: list[RawExcerpt]):
    return canned_function_model(_ExtractorOutput(excerpts=candidates))


def test_async_entrypoint_matches_sync_result_for_same_stub(tmp_path: Path) -> None:
    # The async variant is a wire swap, not a behavior change. For an
    # identical chunk/query/stub it must produce the same ExtractResult as
    # the sync version — same prompt, same raw_candidates, same verified.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")
    doc = normalize(md)
    chunk = chunk_document(doc)[0]
    query = Query(question="?")
    candidates = [RawExcerpt(text="quick brown fox", confidence=0.9)]

    sync_agent = build_extractor_agent()
    async_agent = build_extractor_agent()

    with sync_agent.override(model=_canned(candidates)):
        sync_result = extract_excerpts_verbose(chunk, query, doc, agent=sync_agent)

    async def _run():
        with async_agent.override(model=_canned(candidates)):
            return await extract_excerpts_verbose_async(
                chunk, query, doc, agent=async_agent
            )

    async_result = asyncio.run(_run())

    assert async_result.prompt == sync_result.prompt
    assert async_result.raw_candidates == sync_result.raw_candidates
    assert async_result.verified == sync_result.verified


def test_async_entrypoint_drops_unverified_candidates(tmp_path: Path) -> None:
    # The verification step (substring match in the chunk) must still apply
    # on the async path. A hallucinated span never reaches `verified`.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox.\n")
    doc = normalize(md)
    chunk = chunk_document(doc)[0]

    agent = build_extractor_agent()
    candidates = [
        RawExcerpt(text="quick brown fox"),
        RawExcerpt(text="HALLUCINATED phrase not in doc"),
    ]

    async def _run():
        with agent.override(model=_canned(candidates)):
            return await extract_excerpts_verbose_async(
                chunk, Query(question="?"), doc, agent=agent
            )

    result = asyncio.run(_run())

    assert [c.text for c in result.raw_candidates] == [
        "quick brown fox",
        "HALLUCINATED phrase not in doc",
    ]
    assert [e.text for e in result.verified] == ["quick brown fox"]
