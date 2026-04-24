"""Shared test helpers for stubbing pydantic-ai agents via FunctionModel."""
from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel


def canned_function_model(output: BaseModel) -> FunctionModel:
    """Return a FunctionModel that always produces `output` (serialized as JSON)."""
    payload = output.model_dump_json()

    def fn(messages, info):
        return ModelResponse(parts=[TextPart(content=payload)])

    return FunctionModel(fn)


def varying_function_model(outputs: list[BaseModel]) -> FunctionModel:
    """FunctionModel that cycles through `outputs` — one per call.

    Useful for proving the agent is actually being invoked N times when the
    test assertion would otherwise see identical output each call.
    """
    payloads = [o.model_dump_json() for o in outputs]
    state = {"i": 0}

    def fn(messages, info):
        payload = payloads[state["i"] % len(payloads)]
        state["i"] += 1
        return ModelResponse(parts=[TextPart(content=payload)])

    return FunctionModel(fn)
