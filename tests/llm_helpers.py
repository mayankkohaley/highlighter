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
