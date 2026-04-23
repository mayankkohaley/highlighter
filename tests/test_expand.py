from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from highlighter.expand import _QueryExpansion, build_query_agent, expand_query


def _canned(expansion: _QueryExpansion) -> FunctionModel:
    def fn(messages, info):
        return ModelResponse(parts=[TextPart(content=expansion.model_dump_json())])

    return FunctionModel(fn)


def test_tracer_expands_a_raw_question_into_a_structured_query() -> None:
    agent = build_query_agent()
    canned = _QueryExpansion(
        sub_questions=["Which AWS services are required?", "What IAM permissions?"],
        rubric="Useful excerpts name specific software, permissions, or accounts.",
    )

    with agent.override(model=_canned(canned)):
        query = expand_query("What do I need to deploy an AgentCore agent?", agent=agent)

    assert query.sub_questions == [
        "Which AWS services are required?",
        "What IAM permissions?",
    ]
    assert query.rubric == "Useful excerpts name specific software, permissions, or accounts."


def test_input_question_passes_through_verbatim() -> None:
    agent = build_query_agent()
    # Canned expansion deliberately contains no trace of the input question;
    # expand_query must preserve the raw input in Query.question regardless.
    canned = _QueryExpansion(sub_questions=[], rubric="")

    with agent.override(model=_canned(canned)):
        query = expand_query("  What specifically counts as 'deploy-ready'?  ", agent=agent)

    assert query.question == "  What specifically counts as 'deploy-ready'?  "
