from highlighter.extract import Excerpt
from highlighter.query import Query
from highlighter.synthesize import Synthesis, build_synthesis_agent, synthesize
from tests.llm_helpers import canned_function_model


def _excerpt(text: str, line_start: int, line_end: int) -> Excerpt:
    return Excerpt(
        text=text,
        line_start=line_start,
        line_end=line_end,
        section_path=["S"],
    )


def test_synthesize_with_no_excerpts_short_circuits_to_empty_answer() -> None:
    # An empty excerpt list means nothing verifiable to say. Don't burn tokens
    # asking the model — return a Synthesis that names the gap.
    result = synthesize(Query(question="anything?"), [])

    assert isinstance(result, Synthesis)
    assert result.used_excerpts == []
    assert result.answer


def test_synthesize_returns_structured_answer_from_stubbed_agent() -> None:
    excerpts = [
        _excerpt("alpha line", line_start=3, line_end=3),
        _excerpt("beta line", line_start=5, line_end=5),
    ]
    canned = Synthesis(
        answer="Alpha is described in [1] and beta in [2].",
        used_excerpts=[1, 2],
    )
    agent = build_synthesis_agent()

    with agent.override(model=canned_function_model(canned)):
        result = synthesize(
            Query(question="what's described?"),
            excerpts,
            agent=agent,
        )

    assert result.answer == "Alpha is described in [1] and beta in [2]."
    assert result.used_excerpts == [1, 2]


def test_synthesize_prompt_numbers_excerpts_and_includes_citations() -> None:
    # The model can only cite `[N]` reliably if the prompt actually numbers
    # excerpts and shows each one's text and citation. Capture what we send.
    from pydantic_ai import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    captured: dict[str, str] = {}

    def fn(messages, info):
        # Last user message is the prompt assembled by synthesize().
        captured["prompt"] = messages[-1].parts[-1].content
        return ModelResponse(parts=[TextPart(
            content=Synthesis(answer="ok", used_excerpts=[]).model_dump_json()
        )])

    excerpts = [
        _excerpt("alpha line", line_start=3, line_end=3),
        _excerpt("beta line", line_start=5, line_end=5),
    ]
    agent = build_synthesis_agent()

    with agent.override(model=FunctionModel(fn)):
        synthesize(Query(question="q"), excerpts, agent=agent)

    prompt = captured["prompt"]
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "alpha line" in prompt
    assert "beta line" in prompt
    assert "L3-3" in prompt
    assert "L5-5" in prompt
