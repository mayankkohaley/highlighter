import asyncio
import time
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from highlighter.chunk import chunk_document
from highlighter.expand import _QueryExpansion, build_query_agent
from highlighter.extract import (
    RawExcerpt,
    _ExtractorOutput,
    build_extractor_agent,
)
from highlighter.normalize import normalize
from highlighter.run import run_pipeline
from highlighter.synthesize import Synthesis, build_synthesis_agent
from tests.llm_helpers import canned_function_model


def test_tracer_runs_full_pipeline(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox jumps over the lazy dog.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(
        sub_questions=["What animal is described?"],
        rubric="Useful excerpts name an animal.",
    )
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(
                text="quick brown fox",
                which_subquestion="What animal is described?",
                confidence=0.9,
            )
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="What animal is described?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert result.query.question == "What animal is described?"
    assert result.query.sub_questions == ["What animal is described?"]
    assert len(result.excerpts) == 1
    assert result.excerpts[0].text == "quick brown fox"


def test_excerpts_aggregate_across_all_chunks(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    body = "\n\n".join(f"Filler paragraph {i}." for i in range(40))
    md.write_text(
        "# Title\n\n"
        "MARKER_ALPHA appears early.\n\n"
        f"{body}\n\n"
        "MARKER_BETA appears late.\n"
    )

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="MARKER_ALPHA appears early."),
            RawExcerpt(text="MARKER_BETA appears late."),
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="?",
            chunk_size=80,
            chunk_overlap=10,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    texts = {e.text for e in result.excerpts}
    assert "MARKER_ALPHA appears early." in texts
    assert "MARKER_BETA appears late." in texts


def test_pipeline_result_carries_raw_candidates_including_dropped(tmp_path: Path) -> None:
    # Diagnosing the extractor requires seeing every span the model returned,
    # including ones that failed substring verification. The pipeline result
    # must surface those pre-verification candidates.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nThe quick brown fox.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    # One span that will verify; one that won't.
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="quick brown fox"),
            RawExcerpt(text="HALLUCINATED phrase not in doc"),
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    raw_texts = [c.text for c in result.raw_candidates]
    assert "quick brown fox" in raw_texts
    assert "HALLUCINATED phrase not in doc" in raw_texts
    # Verification still drops the hallucination from excerpts.
    assert [e.text for e in result.excerpts] == ["quick brown fox"]


def test_pipeline_consolidates_adjacent_verified_excerpts(tmp_path: Path) -> None:
    # Stage 4 runs inside the pipeline: two adjacent excerpts in the same
    # section collapse to one consolidated span. The raw verified list is
    # preserved for downstream inspection.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nalpha line\nbeta line\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(
        excerpts=[
            RawExcerpt(text="alpha line"),
            RawExcerpt(text="beta line"),
        ]
    )

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert len(result.excerpts) == 2
    assert len(result.consolidated) == 1
    assert result.consolidated[0].text == "alpha line\nbeta line"
    assert result.consolidated[0].line_start == 3
    assert result.consolidated[0].line_end == 4


def test_pipeline_skips_synthesis_by_default(tmp_path: Path) -> None:
    # Synthesis costs tokens — default is opt-in.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nalpha line.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="alpha line.")])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        result = run_pipeline(
            md,
            question="?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert result.synthesis is None


def test_pipeline_runs_synthesis_when_opted_in(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nalpha line.\n")

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    synthesis_agent = build_synthesis_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    extraction = _ExtractorOutput(excerpts=[RawExcerpt(text="alpha line.")])
    canned_answer = Synthesis(answer="It is alpha. [1]", used_excerpts=[1])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=canned_function_model(extraction)),
        synthesis_agent.override(model=canned_function_model(canned_answer)),
    ):
        result = run_pipeline(
            md,
            question="?",
            synthesize=True,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
            synthesis_agent=synthesis_agent,
        )

    assert result.synthesis is not None
    assert result.synthesis.answer == "It is alpha. [1]"
    assert result.synthesis.used_excerpts == [1]


def test_per_chunk_extraction_runs_concurrently(tmp_path: Path) -> None:
    # Stub each per-chunk extract call to async-sleep 0.3s. With ≥3 chunks
    # and a concurrency cap at or above that, total wall-time should be
    # close to 0.3s, not ~0.9s+. Wide margin so it doesn't flake under load.
    md = tmp_path / "doc.md"
    body = "\n\n".join(f"Filler paragraph {i}." for i in range(40))
    md.write_text(
        "# Title\n\n"
        "MARKER_A appears early.\n\n"
        f"{body}\n\n"
        "MARKER_B somewhere in the middle.\n\n"
        f"{body}\n\n"
        "MARKER_C near the end.\n"
    )
    # Guard: the test has teeth only if the doc actually produces several
    # chunks. If chunking behavior shifts, fail loudly rather than silently
    # weaken the concurrency check.
    assert len(chunk_document(normalize(md), chunk_size=80, chunk_overlap=10)) >= 3

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    payload = _ExtractorOutput(
        excerpts=[RawExcerpt(text="irrelevant")]
    ).model_dump_json()

    async def slow_fn(messages, info):
        await asyncio.sleep(0.3)
        return ModelResponse(parts=[TextPart(content=payload)])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=FunctionModel(slow_fn)),
    ):
        start = time.perf_counter()
        run_pipeline(
            md,
            question="?",
            chunk_size=80,
            chunk_overlap=10,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )
        elapsed = time.perf_counter() - start

    # Sequential would be (n_chunks * 0.3)s ≥ 0.9s. Concurrent should be ~0.3s.
    assert elapsed < 0.7, (
        f"expected concurrent per-chunk extraction, elapsed={elapsed:.2f}s"
    )


def test_excerpts_stay_in_document_order_even_when_completion_order_differs(
    tmp_path: Path,
) -> None:
    # Slower response for earlier chunks forces the later chunks to finish
    # first. `asyncio.gather` must still return results in input order so
    # the excerpt list mirrors document order.
    md = tmp_path / "doc.md"
    body = "\n\n".join(f"Filler paragraph {i}." for i in range(40))
    md.write_text(
        "# Title\n\n"
        "MARKER_A appears early.\n\n"
        f"{body}\n\n"
        "MARKER_B in the middle.\n\n"
        f"{body}\n\n"
        "MARKER_C near the end.\n"
    )
    doc = normalize(md)
    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 3

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")

    def _payload(marker: str) -> str:
        return _ExtractorOutput(
            excerpts=[RawExcerpt(text=marker)]
        ).model_dump_json()

    async def varying_fn(messages, info):
        # Messages end with the user prompt; find which marker is in it and
        # sleep inversely to document position so early chunks finish last.
        prompt = messages[-1].parts[-1].content
        if "MARKER_A" in prompt:
            await asyncio.sleep(0.3)
            return ModelResponse(parts=[TextPart(content=_payload("MARKER_A appears early."))])
        if "MARKER_B" in prompt:
            await asyncio.sleep(0.2)
            return ModelResponse(parts=[TextPart(content=_payload("MARKER_B in the middle."))])
        if "MARKER_C" in prompt:
            await asyncio.sleep(0.1)
            return ModelResponse(parts=[TextPart(content=_payload("MARKER_C near the end."))])
        # Chunks with no marker: return nothing, fast.
        empty = _ExtractorOutput(excerpts=[]).model_dump_json()
        return ModelResponse(parts=[TextPart(content=empty)])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=FunctionModel(varying_fn)),
    ):
        result = run_pipeline(
            md,
            question="?",
            chunk_size=80,
            chunk_overlap=10,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    texts = [e.text for e in result.excerpts]
    # Document order: A before B before C, regardless of which stub finished first.
    assert texts.index("MARKER_A appears early.") < texts.index("MARKER_B in the middle.")
    assert texts.index("MARKER_B in the middle.") < texts.index("MARKER_C near the end.")


def test_per_chunk_exception_does_not_kill_pipeline(tmp_path: Path) -> None:
    # If one chunk's extract call fails, we still want excerpts from the
    # chunks that succeeded. The whole pipeline shouldn't die over a single
    # flaky LLM call; partial highlights are more useful than nothing.
    md = tmp_path / "doc.md"
    body = "\n\n".join(f"Filler paragraph {i}." for i in range(40))
    md.write_text(
        "# Title\n\n"
        "MARKER_A appears early.\n\n"
        f"{body}\n\n"
        "MARKER_B in the middle.\n\n"
        f"{body}\n\n"
        "MARKER_C near the end.\n"
    )

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")

    def _payload(marker: str) -> str:
        return _ExtractorOutput(
            excerpts=[RawExcerpt(text=marker)]
        ).model_dump_json()

    async def flaky_fn(messages, info):
        prompt = messages[-1].parts[-1].content
        if "MARKER_B" in prompt:
            raise RuntimeError("simulated extract failure on MARKER_B chunk")
        if "MARKER_A" in prompt:
            return ModelResponse(parts=[TextPart(content=_payload("MARKER_A appears early."))])
        if "MARKER_C" in prompt:
            return ModelResponse(parts=[TextPart(content=_payload("MARKER_C near the end."))])
        empty = _ExtractorOutput(excerpts=[]).model_dump_json()
        return ModelResponse(parts=[TextPart(content=empty)])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=FunctionModel(flaky_fn)),
    ):
        result = run_pipeline(
            md,
            question="?",
            chunk_size=80,
            chunk_overlap=10,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    texts = {e.text for e in result.excerpts}
    assert "MARKER_A appears early." in texts
    assert "MARKER_C near the end." in texts
    assert "MARKER_B in the middle." not in texts


def test_expand_runs_concurrently_with_chunking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Expand and chunk are independent — overlapping them removes expand's
    # latency from the critical path. With expand stubbed to await 0.3s and
    # chunk being CPU work that completes in milliseconds, chunk_document
    # must finish well before expand_query when the two run concurrently.
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\nbody.\n")

    expand_finished_at: list[float] = []
    chunk_finished_at: list[float] = []

    expansion_payload = _QueryExpansion(sub_questions=[], rubric="").model_dump_json()

    async def slow_expand_fn(messages, info):
        await asyncio.sleep(0.3)
        expand_finished_at.append(time.perf_counter())
        return ModelResponse(parts=[TextPart(content=expansion_payload)])

    import highlighter.run as run_mod

    original_chunk = run_mod.chunk_document

    def tracking_chunk(*args, **kwargs):
        result = original_chunk(*args, **kwargs)
        chunk_finished_at.append(time.perf_counter())
        return result

    monkeypatch.setattr(run_mod, "chunk_document", tracking_chunk)

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    extraction = _ExtractorOutput(excerpts=[])

    with (
        expand_agent.override(model=FunctionModel(slow_expand_fn)),
        extract_agent.override(model=canned_function_model(extraction)),
    ):
        run_pipeline(
            md,
            question="?",
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert chunk_finished_at and expand_finished_at
    assert chunk_finished_at[0] < expand_finished_at[0], (
        "chunking should overlap with in-flight expand_query; "
        f"chunk_done={chunk_finished_at[0]:.3f} expand_done={expand_finished_at[0]:.3f}"
    )


def test_concurrent_extraction_respects_ceiling(tmp_path: Path) -> None:
    # A doc with >32 chunks must never run more than 32 per-chunk extracts at
    # the same time. Rate-limit churn risk scales linearly with fan-out —
    # the ceiling is the guardrail, and this test pins it.
    md = tmp_path / "doc.md"
    body = "\n\n".join(f"Paragraph {i} " + "word " * 20 for i in range(150))
    md.write_text(f"# Title\n\n{body}\n")
    chunk_count = len(chunk_document(normalize(md), chunk_size=80, chunk_overlap=10))
    assert chunk_count > 32, f"test needs >32 chunks to exercise the cap, got {chunk_count}"

    expand_agent = build_query_agent()
    extract_agent = build_extractor_agent()
    expansion = _QueryExpansion(sub_questions=[], rubric="")
    empty_payload = _ExtractorOutput(excerpts=[]).model_dump_json()

    state = {"active": 0, "peak": 0}

    async def tracking_fn(messages, info):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        # Hold long enough that later tasks stack up if the semaphore permits.
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return ModelResponse(parts=[TextPart(content=empty_payload)])

    with (
        expand_agent.override(model=canned_function_model(expansion)),
        extract_agent.override(model=FunctionModel(tracking_fn)),
    ):
        run_pipeline(
            md,
            question="?",
            chunk_size=80,
            chunk_overlap=10,
            expand_agent=expand_agent,
            extract_agent=extract_agent,
        )

    assert state["peak"] <= 32, f"peak concurrency {state['peak']} exceeded ceiling of 32"
