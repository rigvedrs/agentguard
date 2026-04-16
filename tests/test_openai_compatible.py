"""Tests for OpenAI-compatible provider integration."""

from __future__ import annotations

import json
import os

import pytest

from agentguard import GuardConfig, guard
from agentguard.core.registry import global_registry
from agentguard.integrations.openai_compatible import (
    Provider,
    Providers,
    guard_tools,
)


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    global_registry.clear()


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------


class TestProviders:
    def test_all_returns_providers(self):
        """Providers.all() should return a non-empty list of Provider instances."""
        all_providers = Providers.all()
        assert len(all_providers) >= 8  # We defined at least 8 presets
        assert all(isinstance(p, Provider) for p in all_providers)

    def test_by_name_case_insensitive(self):
        """by_name should be case-insensitive and ignore punctuation."""
        assert Providers.by_name("groq") is Providers.GROQ
        assert Providers.by_name("GROQ") is Providers.GROQ
        assert Providers.by_name("OpenRouter") is Providers.OPENROUTER
        assert Providers.by_name("openrouter") is Providers.OPENROUTER
        assert Providers.by_name("Together AI") is Providers.TOGETHER
        assert Providers.by_name("togetherai") is Providers.TOGETHER

    def test_by_name_not_found(self):
        assert Providers.by_name("nonexistent") is None

    def test_provider_client_kwargs(self):
        """client_kwargs should return base_url and api_key."""
        kwargs = Providers.GROQ.client_kwargs(api_key="test-key")
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert kwargs["api_key"] == "test-key"

    def test_provider_with_default_headers(self):
        """OpenRouter should include default_headers."""
        kwargs = Providers.OPENROUTER.client_kwargs(api_key="test")
        assert "default_headers" in kwargs
        assert "HTTP-Referer" in kwargs["default_headers"]

    def test_provider_without_default_headers(self):
        """Groq should NOT include default_headers."""
        kwargs = Providers.GROQ.client_kwargs(api_key="test")
        assert "default_headers" not in kwargs

    def test_provider_env_key_names(self):
        """Verify env key naming follows conventions."""
        assert Providers.OPENAI.env_key == "OPENAI_API_KEY"
        assert Providers.OPENROUTER.env_key == "OPENROUTER_API_KEY"
        assert Providers.GROQ.env_key == "GROQ_API_KEY"
        assert Providers.TOGETHER.env_key == "TOGETHER_API_KEY"
        assert Providers.FIREWORKS.env_key == "FIREWORKS_API_KEY"

    def test_custom_provider(self):
        """Users should be able to create their own provider."""
        custom = Provider(
            name="My Provider",
            base_url="https://api.my-provider.com/v1",
            env_key="MY_PROVIDER_KEY",
            default_model="my-model",
        )
        assert custom.name == "My Provider"
        kwargs = custom.client_kwargs(api_key="sk-123")
        assert kwargs["base_url"] == "https://api.my-provider.com/v1"


# ---------------------------------------------------------------------------
# guard_tools
# ---------------------------------------------------------------------------


class TestGuardTools:
    def test_returns_executor(self):
        """guard_tools should return an OpenAIToolExecutor."""
        def my_tool(x: str) -> str:
            return x.upper()

        executor = guard_tools([my_tool])
        assert len(executor.tools) == 1
        assert executor.tools[0]["function"]["name"] == "my_tool"

    def test_schemas_are_openai_format(self):
        """Generated schemas should follow OpenAI's tool format."""
        def search(query: str, limit: int = 10) -> dict:
            """Search for documents."""
            return {"results": []}

        executor = guard_tools([search])
        schema = executor.tools[0]
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]

    def test_execute_tool_call(self):
        """Executor should dispatch mock tool calls correctly."""
        def add(a: int, b: int) -> dict:
            """Add two numbers."""
            return {"sum": a + b}

        executor = guard_tools([add])

        class MockTC:
            id = "call_1"
            class function:
                name = "add"
                arguments = '{"a": 3, "b": 7}'

        results = executor.execute_all([MockTC()])
        assert len(results) == 1
        data = json.loads(results[0]["content"])
        assert data["sum"] == 10

    def test_with_guard_config(self):
        """Config should be applied to registered tools."""
        def my_tool(x: str) -> str:
            return x

        config = GuardConfig(validate_input=True, max_retries=2)
        executor = guard_tools([my_tool], config=config)
        assert len(executor.tools) == 1

    def test_multiple_tools(self):
        """Should handle multiple tools."""
        def tool_a(x: str) -> str:
            return x

        def tool_b(n: int) -> int:
            return n * 2

        executor = guard_tools([tool_a, tool_b])
        assert len(executor.tools) == 2
        names = {t["function"]["name"] for t in executor.tools}
        assert names == {"tool_a", "tool_b"}

    def test_with_provider(self):
        """Provider param should not affect executor functionality."""
        def my_tool(x: str) -> str:
            return x

        executor = guard_tools([my_tool], provider=Providers.GROQ)
        assert len(executor.tools) == 1
