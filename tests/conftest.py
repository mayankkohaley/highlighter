import os

# pydantic-ai eagerly resolves model strings, which requires the provider key
# even when the test will override the model with FunctionModel / TestModel.
# No real requests are made; a dummy key is enough.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")
