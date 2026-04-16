"""Tests for the AutoGen integration.

All AutoGen-specific objects are mocked so the tests run without autogen
being installed. The integration module's optional import path is exercised
through monkeypatching.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from agentguard import GuardConfig
from agentguard.integrations.autogen_integration import (
    GuardedAutoGenTool,
    guard_autogen_tool,
    guard_autogen_tools,
    register_guarded_tools,
)


# ---------------------------------------------------------------------------
# Plain callables used as tools
# ---------------------------------------------------------------------------


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"results for {query}"


def query_db(sql: str, limit: int = 100) -> list:
    """Query the database and return rows."""
    return [{"id": 1, "sql": sql}]


def no_doc(x: int) -> int:
    pass


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# ---------------------------------------------------------------------------
# GuardedAutoGenTool — construction
# ---------------------------------------------------------------------------


class TestGuardedAutoGenToolConstruction:
    def test_basic_construction(self):
        guarded = GuardedAutoGenTool(search_web)
        assert guarded.name == "search_web"
        assert "Search" in guarded.description

    def test_name_preserved(self):
        guarded = GuardedAutoGenTool(search_web)
        assert guarded.name == "search_web"

    def test_description_from_docstring(self):
        guarded = GuardedAutoGenTool(search_web)
        assert "Search the web" in guarded.description

    def test_description_override(self):
        guarded = GuardedAutoGenTool(search_web, description="Custom description")
        assert guarded.description == "Custom description"

    def test_no_docstring_fallback(self):
        guarded = GuardedAutoGenTool(no_doc)
        assert guarded.description == "Run no_doc"

    def test_with_config(self):
        config = GuardConfig(validate_input=True, max_retries=2)
        guarded = GuardedAutoGenTool(search_web, config=config)
        assert guarded.guarded_fn is not None

    def test_wrapped_attribute(self):
        guarded = GuardedAutoGenTool(search_web)
        assert guarded.__wrapped__ is search_web

    def test_annotations_preserved(self):
        guarded = GuardedAutoGenTool(search_web)
        assert "query" in guarded.__annotations__
        assert "return" in guarded.__annotations__

    def test_docstring_preserved(self):
        guarded = GuardedAutoGenTool(search_web)
        assert guarded.__doc__ == search_web.__doc__


# ---------------------------------------------------------------------------
# GuardedAutoGenTool — execution
# ---------------------------------------------------------------------------


class TestGuardedAutoGenToolExecution:
    def test_call_with_kwargs(self):
        guarded = GuardedAutoGenTool(search_web)
        result = guarded(query="Python tutorials")
        assert "Python tutorials" in result

    def test_call_with_positional(self):
        guarded = GuardedAutoGenTool(search_web)
        result = guarded("positional query")
        assert "positional query" in result

    def test_returns_correct_type(self):
        guarded = GuardedAutoGenTool(multiply)
        result = guarded(a=3.0, b=4.0)
        assert result == 12.0

    def test_default_args_honored(self):
        guarded = GuardedAutoGenTool(query_db)
        result = guarded(sql="SELECT 1")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_acall(self):
        """acall method should exist and be callable on the guarded tool."""
        guarded = GuardedAutoGenTool(search_web)
        # acall is present on GuardedAutoGenTool
        assert callable(guarded.acall)

    def test_validation_input_failure(self):
        """Type validation errors should propagate."""

        def strict_fn(count: int) -> int:
            return count * 2

        config = GuardConfig(validate_input=True)
        guarded = GuardedAutoGenTool(strict_fn, config=config)
        with pytest.raises(Exception):
            guarded(count="not_an_int")


# ---------------------------------------------------------------------------
# GuardedAutoGenTool — as_function
# ---------------------------------------------------------------------------


class TestAsFunction:
    def test_as_function_callable(self):
        guarded = GuardedAutoGenTool(search_web)
        proxy = guarded.as_function()
        assert callable(proxy)

    def test_as_function_executes(self):
        guarded = GuardedAutoGenTool(search_web)
        proxy = guarded.as_function()
        result = proxy(query="via proxy")
        assert "via proxy" in result

    def test_as_function_name(self):
        guarded = GuardedAutoGenTool(search_web)
        proxy = guarded.as_function()
        assert proxy.__name__ == "search_web"

    def test_as_function_doc(self):
        guarded = GuardedAutoGenTool(search_web)
        proxy = guarded.as_function()
        assert proxy.__doc__ == search_web.__doc__


# ---------------------------------------------------------------------------
# GuardedAutoGenTool — register
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_raises_without_autogen(self):
        with patch(
            "agentguard.integrations.autogen_integration._AUTOGEN_AVAILABLE", False
        ):
            guarded = GuardedAutoGenTool(search_web)
            with pytest.raises(ImportError, match="autogen is not installed"):
                guarded.register(MagicMock(), MagicMock())

    def test_register_calls_agent_methods(self):
        """register() must call register_for_llm and register_for_execution."""
        with patch(
            "agentguard.integrations.autogen_integration._AUTOGEN_AVAILABLE", True
        ):
            llm_agent = MagicMock()
            executor_agent = MagicMock()

            # Set up the mock chain: register_for_execution returns a decorator
            # that returns a proxy; register_for_llm does the same.
            executor_agent.register_for_execution.return_value = lambda fn: fn
            llm_agent.register_for_llm.return_value = lambda fn: fn

            guarded = GuardedAutoGenTool(search_web)
            guarded.register(llm_agent, executor_agent)

            executor_agent.register_for_execution.assert_called_once()
            llm_agent.register_for_llm.assert_called_once_with(
                description=guarded.description
            )

    def test_register_uses_custom_description(self):
        with patch(
            "agentguard.integrations.autogen_integration._AUTOGEN_AVAILABLE", True
        ):
            llm_agent = MagicMock()
            executor_agent = MagicMock()
            executor_agent.register_for_execution.return_value = lambda fn: fn
            llm_agent.register_for_llm.return_value = lambda fn: fn

            guarded = GuardedAutoGenTool(search_web)
            guarded.register(llm_agent, executor_agent, description="Custom desc")

            llm_agent.register_for_llm.assert_called_once_with(description="Custom desc")


# ---------------------------------------------------------------------------
# @guard_autogen_tool decorator
# ---------------------------------------------------------------------------


class TestGuardAutoGenToolDecorator:
    def test_no_parens(self):
        @guard_autogen_tool
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert isinstance(my_tool, GuardedAutoGenTool)
        assert my_tool.name == "my_tool"

    def test_with_config_parens(self):
        config = GuardConfig(validate_input=True, max_retries=1)

        @guard_autogen_tool(config=config)
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert isinstance(my_tool, GuardedAutoGenTool)

    def test_with_description_override(self):
        @guard_autogen_tool(description="Override description")
        def my_tool(x: str) -> str:
            """Original docstring."""
            return x

        assert my_tool.description == "Override description"

    def test_returns_correct_result(self):
        @guard_autogen_tool
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        assert double(n=5) == 10

    def test_empty_parens(self):
        @guard_autogen_tool()
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert isinstance(my_tool, GuardedAutoGenTool)


# ---------------------------------------------------------------------------
# guard_autogen_tools — bulk wrapper
# ---------------------------------------------------------------------------


class TestGuardAutoGenTools:
    def test_returns_dict(self):
        result = guard_autogen_tools([search_web, query_db])
        assert isinstance(result, dict)
        assert "search_web" in result
        assert "query_db" in result

    def test_all_are_guarded_tools(self):
        result = guard_autogen_tools([search_web, query_db])
        for tool in result.values():
            assert isinstance(tool, GuardedAutoGenTool)

    def test_empty_list(self):
        result = guard_autogen_tools([])
        assert result == {}

    def test_config_applied(self):
        config = GuardConfig(validate_input=True, max_retries=2)
        result = guard_autogen_tools([search_web], config=config)
        assert result["search_web"].guarded_fn is not None

    def test_execution_from_dict(self):
        result = guard_autogen_tools([search_web])
        output = result["search_web"](query="dict call")
        assert "dict call" in output


# ---------------------------------------------------------------------------
# register_guarded_tools
# ---------------------------------------------------------------------------


class TestRegisterGuardedTools:
    def test_raises_without_autogen(self):
        with patch(
            "agentguard.integrations.autogen_integration._AUTOGEN_AVAILABLE", False
        ):
            with pytest.raises(ImportError):
                register_guarded_tools([search_web], MagicMock(), MagicMock())

    def test_returns_dict(self):
        with patch(
            "agentguard.integrations.autogen_integration._AUTOGEN_AVAILABLE", True
        ):
            llm_agent = MagicMock()
            executor_agent = MagicMock()
            executor_agent.register_for_execution.return_value = lambda fn: fn
            llm_agent.register_for_llm.return_value = lambda fn: fn

            result = register_guarded_tools([search_web, query_db], llm_agent, executor_agent)

            assert "search_web" in result
            assert "query_db" in result
            assert all(isinstance(t, GuardedAutoGenTool) for t in result.values())

    def test_all_tools_registered(self):
        with patch(
            "agentguard.integrations.autogen_integration._AUTOGEN_AVAILABLE", True
        ):
            llm_agent = MagicMock()
            executor_agent = MagicMock()
            executor_agent.register_for_execution.return_value = lambda fn: fn
            llm_agent.register_for_llm.return_value = lambda fn: fn

            register_guarded_tools([search_web, query_db], llm_agent, executor_agent)

            # register_for_llm should have been called twice (once per tool)
            assert llm_agent.register_for_llm.call_count == 2
            assert executor_agent.register_for_execution.call_count == 2


# ---------------------------------------------------------------------------
# Integration with agentguard features
# ---------------------------------------------------------------------------


class TestIntegrationFeatures:
    def test_circuit_breaker(self):
        from agentguard.core.types import CircuitBreakerConfig

        config = GuardConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=3)
        )
        guarded = GuardedAutoGenTool(search_web, config=config)
        result = guarded(query="circuit test")
        assert result

    def test_retry_on_failure(self):
        call_count = [0]

        def flaky(query: str) -> str:
            """Flaky tool that fails first 2 calls."""
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("Temporary failure")
            return f"ok: {query}"

        config = GuardConfig(max_retries=3)
        guarded = GuardedAutoGenTool(flaky, config=config)
        result = guarded(query="retry me")
        assert result == "ok: retry me"
        assert call_count[0] == 3

    def test_trace_recording(self, tmp_path):
        config = GuardConfig(record=True, trace_dir=str(tmp_path))
        guarded = GuardedAutoGenTool(search_web, config=config)
        guarded(query="trace test")

        sqlite_trace = tmp_path / "agentguard_traces.db"
        jsonl_traces = list(tmp_path.glob("*.jsonl"))
        assert sqlite_trace.exists() or len(jsonl_traces) > 0

    def test_repr(self):
        guarded = GuardedAutoGenTool(search_web)
        r = repr(guarded)
        assert "GuardedAutoGenTool" in r
        assert "search_web" in r
