"""Tests for framework integrations: OpenAI, Anthropic, LangChain, MCP."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentguard import GuardConfig, guard
from agentguard.core.registry import global_registry
from agentguard.integrations.anthropic_integration import (
    AnthropicToolExecutor,
    function_to_anthropic_tool,
    guard_anthropic_tools,
)
from agentguard.integrations.langchain_integration import (
    GuardedLangChainTool,
    guard_langchain_tools,
)
from agentguard.integrations.mcp_integration import GuardedMCPClient, GuardedMCPServer
from agentguard.integrations.openai_integration import (
    OpenAIToolExecutor,
    function_to_openai_tool,
    guard_openai_tools,
)


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    global_registry.clear()


# ---------------------------------------------------------------------------
# OpenAI integration
# ---------------------------------------------------------------------------


def search_web(query: str) -> dict:
    """Search the web for the given query."""
    return {"results": [query], "count": 1}


def get_weather(city: str, units: str = "metric") -> dict:
    """Get weather for a city."""
    return {"temperature": 20, "city": city, "units": units}


class TestOpenAIIntegration:
    def test_function_to_openai_tool(self):
        schema = function_to_openai_tool(search_web)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_web"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]

    def test_optional_param_not_required(self):
        schema = function_to_openai_tool(get_weather)
        required = schema["function"]["parameters"]["required"]
        assert "city" in required
        assert "units" not in required

    def test_guard_openai_tools_returns_schemas(self):
        schemas = guard_openai_tools([search_web, get_weather])
        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "search_web" in names
        assert "get_weather" in names

    def test_guard_openai_tools_with_config(self):
        config = GuardConfig(validate_input=True, max_retries=1)
        schemas = guard_openai_tools([search_web], config=config)
        assert len(schemas) == 1

    def test_openai_executor_register_and_execute(self):
        executor = OpenAIToolExecutor()
        executor.register(search_web)

        tool_call = MagicMock()
        tool_call.function.name = "search_web"
        tool_call.function.arguments = '{"query": "python tutorials"}'

        result = executor.execute(tool_call)
        import json
        data = json.loads(result)
        assert "results" in data

    def test_openai_executor_unknown_tool(self):
        executor = OpenAIToolExecutor()
        tool_call = MagicMock()
        tool_call.function.name = "nonexistent_tool"
        tool_call.function.arguments = "{}"
        result = executor.execute(tool_call)
        import json
        data = json.loads(result)
        assert "error" in data

    def test_openai_executor_tools_property(self):
        executor = OpenAIToolExecutor()
        executor.register(search_web)
        executor.register(get_weather)
        assert len(executor.tools) == 2

    def test_openai_executor_execute_all(self):
        executor = OpenAIToolExecutor()
        executor.register(search_web)

        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "search_web"
        tc1.function.arguments = '{"query": "test"}'

        results = executor.execute_all([tc1])
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# Anthropic integration
# ---------------------------------------------------------------------------


class TestAnthropicIntegration:
    def test_function_to_anthropic_tool(self):
        schema = function_to_anthropic_tool(search_web)
        assert schema["name"] == "search_web"
        assert "input_schema" in schema
        assert "query" in schema["input_schema"]["properties"]

    def test_guard_anthropic_tools(self):
        schemas = guard_anthropic_tools([search_web, get_weather])
        assert len(schemas) == 2

    def test_anthropic_executor_execute(self):
        executor = AnthropicToolExecutor()
        executor.register(search_web)

        block = MagicMock()
        block.type = "tool_use"
        block.name = "search_web"
        block.id = "tu_1"
        block.input = {"query": "hello"}

        result = executor.execute(block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tu_1"

    def test_anthropic_executor_execute_all(self):
        executor = AnthropicToolExecutor()
        executor.register(get_weather)

        text_block = MagicMock()
        text_block.type = "text"

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "get_weather"
        tool_block.id = "tu_2"
        tool_block.input = {"city": "London"}

        results = executor.execute_all([text_block, tool_block])
        # Only tool_use blocks should be executed
        assert len(results) == 1
        assert results[0]["tool_use_id"] == "tu_2"

    def test_anthropic_executor_error_handling(self):
        def broken(x: str) -> dict:
            raise RuntimeError("always breaks")

        executor = AnthropicToolExecutor()
        executor.register(broken)

        block = MagicMock()
        block.type = "tool_use"
        block.name = "broken"
        block.id = "tu_err"
        block.input = {"x": "test"}

        result = executor.execute(block)
        assert result.get("is_error") is True


# ---------------------------------------------------------------------------
# LangChain integration
# ---------------------------------------------------------------------------


class TestLangChainIntegration:
    def test_guarded_tool_creation(self):
        tool = GuardedLangChainTool.from_function(search_web)
        assert tool.name == "search_web"
        assert "Search" in tool.description or "search" in tool.description.lower()

    def test_guarded_tool_call(self):
        tool = GuardedLangChainTool.from_function(search_web)
        result = tool("python tutorials")
        assert "results" in result

    def test_guarded_tool_run(self):
        tool = GuardedLangChainTool.from_function(get_weather)
        result = tool._run("London")
        assert "temperature" in result

    def test_guard_langchain_tools_bulk(self):
        tools = guard_langchain_tools([search_web, get_weather])
        assert len(tools) == 2
        assert all(isinstance(t, GuardedLangChainTool) for t in tools)

    def test_custom_name_and_description(self):
        tool = GuardedLangChainTool.from_function(
            search_web,
            name="web_search",
            description="Custom description",
        )
        assert tool.name == "web_search"
        assert tool.description == "Custom description"

    def test_openai_function_schema(self):
        tool = GuardedLangChainTool.from_function(search_web)
        schema = tool.to_openai_function()
        assert schema["function"]["name"] == "search_web"

    @pytest.mark.asyncio
    async def test_guarded_tool_arun(self):
        @guard
        async def async_search(query: str) -> dict:
            return {"results": [query]}

        tool = GuardedLangChainTool.from_function(async_search)
        result = await tool._arun("test query")
        assert "results" in result


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    @pytest.mark.asyncio
    async def test_guarded_mcp_server_call_tool(self):
        class FakeMCPServer:
            async def call_tool(self, name: str, arguments: dict) -> dict:
                return {"tool": name, "args": arguments}

            async def list_tools(self):
                return [{"name": "search"}]

        server = GuardedMCPServer(FakeMCPServer(), config=GuardConfig())
        result = await server.call_tool("search", {"query": "test"})
        assert result["tool"] == "search"

    def test_guarded_mcp_server_sync(self):
        class FakeMCPServer:
            def call_tool_sync(self, name: str, arguments: dict) -> dict:
                return {"tool": name}

        server = GuardedMCPServer(FakeMCPServer())
        # The guarded tool should be created on demand
        assert server is not None

    def test_guarded_mcp_server_proxy(self):
        class FakeMCPServer:
            some_attr = "hello"

        server = GuardedMCPServer(FakeMCPServer())
        assert server.some_attr == "hello"

    @pytest.mark.asyncio
    async def test_guarded_mcp_client(self):
        class FakeMCPClient:
            async def call_tool(self, name: str, arguments: dict) -> dict:
                return {"result": "ok"}

        client = GuardedMCPClient(FakeMCPClient(), config=GuardConfig(record=False))
        result = await client.call_tool("my_tool", {"x": 1})
        assert result == {"result": "ok"}
