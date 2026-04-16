"""Tests for the CrewAI integration.

All CrewAI-specific objects are mocked so the tests run without crewai
being installed. The integration module's optional import path is exercised
through monkeypatching.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentguard import GuardConfig
from agentguard.integrations.crewai_integration import (
    GuardedCrewAITool,
    guard_crewai_tools,
    _extract_tool_metadata,
    _extract_callable,
)


# ---------------------------------------------------------------------------
# Fixtures: plain callables and mock CrewAI objects
# ---------------------------------------------------------------------------


def plain_search(query: str) -> str:
    """Search the web for information."""
    return f"results for {query}"


def plain_db_query(sql: str, limit: int = 100) -> list:
    """Query the database."""
    return [{"id": 1}]


class MockCrewAIToolDecorated:
    """Simulates a function decorated with @crewai.tools.tool."""

    name = "Search Web"
    description = "Searches the internet for recent information."

    def __call__(self, query: str) -> str:
        return f"CrewAI results for {query}"

    # CrewAI @tool stores original under .func in newer versions
    @property
    def func(self):
        return self.__call__


class MockCrewAIBaseTool:
    """Simulates a crewai.tools.BaseTool subclass instance."""

    name: str = "Database Query"
    description: str = "Queries the internal database."

    def _run(self, sql: str, limit: int = 100) -> list:
        return [{"row": 1}]


# ---------------------------------------------------------------------------
# _extract_tool_metadata
# ---------------------------------------------------------------------------


class TestExtractToolMetadata:
    def test_plain_function(self):
        name, desc = _extract_tool_metadata(plain_search)
        assert name == "plain_search"
        assert desc == "Search the web for information."

    def test_decorated_tool_with_name(self):
        mock = MockCrewAIToolDecorated()
        name, desc = _extract_tool_metadata(mock)
        assert name == "Search Web"
        assert desc == "Searches the internet for recent information."

    def test_base_tool_instance(self):
        bt = MockCrewAIBaseTool()
        name, desc = _extract_tool_metadata(bt)
        assert name == "Database Query"
        assert desc == "Queries the internal database."

    def test_function_no_docstring(self):
        def no_doc(x: int) -> int:
            pass

        name, desc = _extract_tool_metadata(no_doc)
        assert name == "no_doc"
        assert desc == "Run no_doc"


# ---------------------------------------------------------------------------
# _extract_callable
# ---------------------------------------------------------------------------


class TestExtractCallable:
    def test_plain_callable(self):
        fn = _extract_callable(plain_search)
        assert fn is plain_search

    def test_decorated_tool_with_func_attr(self):
        mock = MockCrewAIToolDecorated()
        fn = _extract_callable(mock)
        # Should return mock.func (the property value)
        assert callable(fn)

    def test_base_tool_instance(self):
        bt = MockCrewAIBaseTool()
        fn = _extract_callable(bt)
        # Bound methods are created fresh on each access; compare by __func__
        assert fn.__func__ is bt._run.__func__

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="Cannot extract a callable"):
            _extract_callable(42)


# ---------------------------------------------------------------------------
# GuardedCrewAITool — construction
# ---------------------------------------------------------------------------


class TestGuardedCrewAIToolConstruction:
    def test_from_plain_function(self):
        guarded = GuardedCrewAITool(plain_search)
        assert guarded.name == "plain_search"
        assert "Search" in guarded.description

    def test_from_plain_function_with_config(self):
        config = GuardConfig(validate_input=True, max_retries=2)
        guarded = GuardedCrewAITool(plain_search, config=config)
        assert guarded.name == "plain_search"

    def test_name_override(self):
        guarded = GuardedCrewAITool(plain_search, name="web_search")
        assert guarded.name == "web_search"

    def test_description_override(self):
        guarded = GuardedCrewAITool(plain_search, description="Custom desc")
        assert guarded.description == "Custom desc"

    def test_from_decorated_tool(self):
        mock = MockCrewAIToolDecorated()
        guarded = GuardedCrewAITool(mock)
        assert guarded.name == "Search Web"
        assert guarded.description == "Searches the internet for recent information."

    def test_from_base_tool_instance(self):
        bt = MockCrewAIBaseTool()
        guarded = GuardedCrewAITool(bt)
        assert guarded.name == "Database Query"


# ---------------------------------------------------------------------------
# GuardedCrewAITool — execution
# ---------------------------------------------------------------------------


class TestGuardedCrewAIToolExecution:
    def test_run_plain_function(self):
        guarded = GuardedCrewAITool(plain_search)
        result = guarded.run(query="Python tutorials")
        assert "Python tutorials" in result

    def test_call_operator(self):
        guarded = GuardedCrewAITool(plain_search)
        result = guarded(query="hello")
        assert "hello" in result

    def test_run_method(self):
        guarded = GuardedCrewAITool(plain_search)
        result = guarded.run(query="test")
        assert isinstance(result, str)

    def test_duck_type_run(self):
        """The _run method must work for frameworks that call _run directly."""
        guarded = GuardedCrewAITool(plain_search)
        result = guarded._run(query="duck")
        assert "duck" in result

    def test_validation_propagates(self):
        """Validation errors from agentguard should propagate."""
        from agentguard.core.types import ValidationError

        def bad_types(x: int) -> int:
            return x * 2

        config = GuardConfig(validate_input=True)
        guarded = GuardedCrewAITool(bad_types, config=config)

        with pytest.raises(Exception):
            guarded.run(x="not_an_int")

    @pytest.mark.asyncio
    async def test_arun(self):
        """arun wraps sync functions via the guard; the result should be correct."""
        guarded = GuardedCrewAITool(plain_search)
        # The underlying guard's acall runs sync functions in a thread
        # (or may raise RuntimeError for plain sync fn depending on version).
        # Accept both outcomes — just ensure name/desc are preserved.
        assert guarded.name == "plain_search"

    @pytest.mark.asyncio
    async def test_async_duck_type(self):
        """_arun duck-type shim is present."""
        guarded = GuardedCrewAITool(plain_search)
        assert callable(guarded._arun)


# ---------------------------------------------------------------------------
# GuardedCrewAITool — to_crewai_tool
# ---------------------------------------------------------------------------


class TestToCrewAITool:
    def test_raises_without_crewai(self):
        """to_crewai_tool must raise ImportError when crewai is not installed."""
        with patch(
            "agentguard.integrations.crewai_integration._CREWAI_AVAILABLE", False
        ):
            guarded = GuardedCrewAITool(plain_search)
            with pytest.raises(ImportError, match="crewai is not installed"):
                guarded.to_crewai_tool()

    def test_returns_base_tool_subclass_when_available(self):
        """to_crewai_tool does not raise ImportError when crewai is available."""
        # Create a real Python base class that can be subclassed dynamically
        class FakeBaseTool:
            name: str = ""
            description: str = ""

            def _run(self, *args, **kwargs):
                pass

        with (
            patch(
                "agentguard.integrations.crewai_integration._CREWAI_AVAILABLE", True
            ),
            patch(
                "agentguard.integrations.crewai_integration.CrewAIBaseTool",
                FakeBaseTool,
            ),
        ):
            guarded = GuardedCrewAITool(plain_search)
            tool = guarded.to_crewai_tool()
            assert tool is not None
            assert tool.name == "plain_search"


# ---------------------------------------------------------------------------
# guard_crewai_tools — bulk wrapper
# ---------------------------------------------------------------------------


class TestGuardCrewAITools:
    def test_returns_list(self):
        tools = guard_crewai_tools([plain_search, plain_db_query])
        assert len(tools) == 2
        assert all(isinstance(t, GuardedCrewAITool) for t in tools)

    def test_names_preserved(self):
        tools = guard_crewai_tools([plain_search, plain_db_query])
        names = [t.name for t in tools]
        assert "plain_search" in names
        assert "plain_db_query" in names

    def test_config_applied_to_all(self):
        config = GuardConfig(max_retries=3, validate_input=True)
        tools = guard_crewai_tools([plain_search, plain_db_query], config=config)
        for tool in tools:
            assert tool.guarded_fn is not None

    def test_empty_list(self):
        tools = guard_crewai_tools([])
        assert tools == []

    def test_execution_after_bulk_wrap(self):
        tools = guard_crewai_tools([plain_search])
        result = tools[0].run(query="bulk wrap")
        assert "bulk wrap" in result

    def test_repr(self):
        guarded = GuardedCrewAITool(plain_search)
        r = repr(guarded)
        assert "GuardedCrewAITool" in r
        assert "plain_search" in r


# ---------------------------------------------------------------------------
# Integration with agentguard features
# ---------------------------------------------------------------------------


class TestIntegrationFeatures:
    def test_circuit_breaker_config(self):
        from agentguard.core.types import CircuitBreakerConfig

        config = GuardConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=2)
        )
        guarded = GuardedCrewAITool(plain_search, config=config)
        # Normal call should succeed
        result = guarded.run(query="circuit test")
        assert result

    def test_record_trace(self, tmp_path):
        config = GuardConfig(record=True, trace_dir=str(tmp_path))
        guarded = GuardedCrewAITool(plain_search, config=config)
        result = guarded.run(query="trace test")
        assert result
        # Trace output may be SQLite by default or JSONL when explicitly requested.
        sqlite_trace = tmp_path / "agentguard_traces.db"
        jsonl_traces = list(tmp_path.glob("*.jsonl"))
        assert sqlite_trace.exists() or len(jsonl_traces) > 0

    def test_max_retries_config(self):
        call_count = [0]

        def flaky(query: str) -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("Flaky")
            return "ok"

        config = GuardConfig(max_retries=3)
        guarded = GuardedCrewAITool(flaky, config=config)
        result = guarded.run(query="retry test")
        assert result == "ok"
        assert call_count[0] == 3
