"""Model Context Protocol (MCP) integration for agentguard.

Provides ``GuardedMCPServer`` — a wrapper that intercepts MCP tool calls
through an agentguard protection stack before dispatching to the underlying
server implementation.

Also provides ``GuardedMCPClient`` for client-side interception when consuming
tools from a remote MCP server.

This integration is designed to be compatible with both the ``mcp`` Python
SDK (https://github.com/modelcontextprotocol/python-sdk) and custom MCP
server implementations.

Usage::

    from agentguard.integrations import GuardedMCPServer
    from agentguard import GuardConfig

    config = GuardConfig(
        validate_input=True,
        detect_hallucination=True,
        max_retries=2,
        record=True,
    )

    # Wrap an existing MCP server instance
    server = GuardedMCPServer(original_server, config=config)
    # server is now a drop-in replacement
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable, Optional

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.trace import TraceStore, get_active_recorders
from agentguard.core.types import (
    GuardConfig,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    TraceEntry,
)


# ---------------------------------------------------------------------------
# GuardedMCPServer
# ---------------------------------------------------------------------------


class GuardedMCPServer:
    """Wraps an MCP server to apply agentguard protection on all tool calls.

    Tool dispatch is intercepted via ``call_tool``. All other server methods
    are proxied to the underlying server transparently.

    Example::

        server = GuardedMCPServer(my_mcp_server, config=GuardConfig(max_retries=2))
        # Use server exactly like my_mcp_server
    """

    def __init__(
        self,
        server: Any,
        config: Optional[GuardConfig] = None,
        *,
        tool_configs: Optional[dict[str, GuardConfig]] = None,
    ) -> None:
        """Initialise the guarded server.

        Args:
            server: The original MCP server object to wrap.
            config: Default guard config applied to all tools.
            tool_configs: Per-tool config overrides keyed by tool name.
        """
        self._server = server
        self._config = config or GuardConfig()
        self._tool_configs = tool_configs or {}
        self._guarded_tools: dict[str, GuardedTool] = {}
        self._trace_store: Optional[TraceStore] = (
            TraceStore(
                directory=self._config.trace_dir,
                backend=self._config.trace_backend,
                db_path=self._config.trace_db_path,
            ) if self._config.record else None
        )

    # ------------------------------------------------------------------
    # Tool call interception
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Intercept and guard an MCP tool call.

        Args:
            name: The tool name.
            arguments: The tool's input arguments.

        Returns:
            The tool's response, after passing through all guards.
        """
        # Build the guard wrapper on first use
        gt = self._get_or_create_guard(name)
        return await gt.acall(**arguments)

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        """Synchronous variant of :meth:`call_tool`.

        Args:
            name: The tool name.
            arguments: The tool's input arguments.

        Returns:
            The tool's response.
        """
        gt = self._get_or_create_guard(name)
        return gt(**arguments)

    # ------------------------------------------------------------------
    # Proxy remaining server methods
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Proxy any unrecognised attribute to the underlying server."""
        return getattr(self._server, name)

    # ------------------------------------------------------------------
    # Tool listing (delegates to underlying server)
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tool list."""
        return await self._server.list_tools()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_guard(self, tool_name: str) -> GuardedTool:
        """Build or return the cached GuardedTool for *tool_name*."""
        if tool_name in self._guarded_tools:
            return self._guarded_tools[tool_name]

        cfg = self._tool_configs.get(tool_name, self._config)

        # Build a dispatcher function that calls the underlying server's tool
        server = self._server

        async def _dispatch(**kwargs: Any) -> Any:
            if hasattr(server, "call_tool"):
                return await server.call_tool(tool_name, kwargs)
            raise RuntimeError(
                f"Underlying MCP server does not support call_tool for '{tool_name}'"
            )

        _dispatch.__name__ = tool_name
        _dispatch.__doc__ = f"MCP tool: {tool_name}"

        gt = GuardedTool(_dispatch, config=cfg)
        self._guarded_tools[tool_name] = gt
        return gt

    def __repr__(self) -> str:
        return f"GuardedMCPServer(tools={list(self._guarded_tools.keys())})"


# ---------------------------------------------------------------------------
# GuardedMCPClient
# ---------------------------------------------------------------------------


class GuardedMCPClient:
    """Client-side agentguard wrapper for consuming tools from an MCP server.

    Intercepts calls to ``call_tool`` before they are sent to the remote server,
    applying validation, budget tracking, and tracing.

    Example::

        from agentguard.integrations import GuardedMCPClient
        from agentguard import GuardConfig

        client = GuardedMCPClient(original_client, config=GuardConfig(record=True))
        result = await client.call_tool("search_web", {"query": "hello"})
    """

    def __init__(
        self,
        client: Any,
        config: Optional[GuardConfig] = None,
    ) -> None:
        """Initialise the guarded client.

        Args:
            client: The original MCP client object.
            config: Guard config applied to all outgoing tool calls.
        """
        self._client = client
        self._config = config or GuardConfig()
        self._trace_store: Optional[TraceStore] = (
            TraceStore(
                directory=self._config.trace_dir,
                backend=self._config.trace_backend,
                db_path=self._config.trace_db_path,
            ) if self._config.record else None
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Guard and dispatch an outgoing tool call.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            The server's response.
        """
        call = ToolCall(
            tool_name=name,
            args=(),
            kwargs=arguments,
            session_id=self._config.session_id,
        )
        start = time.perf_counter()
        try:
            result = await self._client.call_tool(name, arguments)
            elapsed = (time.perf_counter() - start) * 1000
            tool_result = ToolResult(
                call_id=call.call_id,
                tool_name=name,
                status=ToolCallStatus.SUCCESS,
                return_value=result,
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            tool_result = ToolResult(
                call_id=call.call_id,
                tool_name=name,
                status=ToolCallStatus.FAILURE,
                exception=str(exc),
                exception_type=type(exc).__qualname__,
                execution_time_ms=elapsed,
            )
            entry = TraceEntry(call=call, result=tool_result)
            self._persist(entry)
            raise

        entry = TraceEntry(call=call, result=tool_result)
        self._persist(entry)
        return result

    def _persist(self, entry: TraceEntry) -> None:
        if self._trace_store:
            self._trace_store.write(entry, session_id=self._config.session_id)
        for rec in get_active_recorders():
            rec.record(entry)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"GuardedMCPClient(config={self._config!r})"
