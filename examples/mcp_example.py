"""MCP (Model Context Protocol) integration example for agentguard.

Shows how to protect MCP server tools and client calls with
agentguard guardrails, tracing, and hallucination detection.

Run with: python examples/mcp_example.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from agentguard import GuardConfig
from agentguard.integrations import GuardedMCPServer, GuardedMCPClient


# ---------------------------------------------------------------------------
# Mock MCP server (simulates a real MCP server)
# ---------------------------------------------------------------------------


class MockMCPServer:
    """A minimal MCP server stub for demonstration."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call."""
        if name == "search_docs":
            await asyncio.sleep(0.02)
            return {
                "results": [
                    {"title": f"Doc about {arguments.get('query', '')}", "score": 0.95},
                ],
                "total": 1,
            }
        if name == "run_sql":
            await asyncio.sleep(0.01)
            return {"rows": [{"id": 1, "value": "example"}], "count": 1}
        return {"error": f"Unknown tool: {name}"}

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "search_docs", "description": "Search the document store"},
            {"name": "run_sql", "description": "Execute a SQL query"},
        ]


# ---------------------------------------------------------------------------
# Guarded MCP server demo
# ---------------------------------------------------------------------------


async def demo_guarded_server() -> None:
    """Wrap an MCP server with agentguard."""
    print("\n--- Guarded MCP Server ---")

    config = GuardConfig(
        validate_input=True,
        max_retries=2,
        timeout=10.0,
        record=True,
        trace_dir="/tmp/agentguard_mcp",
    )

    server = MockMCPServer()
    guarded = GuardedMCPServer(server, config=config)

    # List available tools (delegates to underlying server)
    tools = await guarded.list_tools()
    print(f"  Available tools: {[t['name'] for t in tools]}")

    # Execute a guarded tool call
    result = await guarded.call_tool("search_docs", {"query": "agentguard"})
    print(f"  search_docs result: {json.dumps(result, indent=2)[:100]}")

    result = await guarded.call_tool("run_sql", {"query": "SELECT 1"})
    print(f"  run_sql result: {json.dumps(result, indent=2)[:80]}")


# ---------------------------------------------------------------------------
# Guarded MCP client demo
# ---------------------------------------------------------------------------


async def demo_guarded_client() -> None:
    """Wrap an MCP client with agentguard."""
    print("\n--- Guarded MCP Client ---")

    config = GuardConfig(record=True, trace_dir="/tmp/agentguard_mcp")
    server = MockMCPServer()  # In real usage, this is a connected MCP client
    client = GuardedMCPClient(server, config=config)

    result = await client.call_tool("search_docs", {"query": "Python SDK"})
    print(f"  Client call result: {json.dumps(result, indent=2)[:100]}")


# ---------------------------------------------------------------------------
# Per-tool config overrides
# ---------------------------------------------------------------------------


async def demo_per_tool_config() -> None:
    """Show per-tool configuration overrides."""
    print("\n--- Per-Tool Config ---")

    base_config = GuardConfig(validate_input=True, max_retries=1)
    sql_config = GuardConfig(validate_input=True, max_retries=3, timeout=30.0)

    server = MockMCPServer()
    guarded = GuardedMCPServer(
        server,
        config=base_config,
        tool_configs={"run_sql": sql_config},
    )

    # search_docs uses base_config (1 retry)
    result = await guarded.call_tool("search_docs", {"query": "test"})
    print(f"  search_docs (base config): {result}")

    # run_sql uses sql_config (3 retries, 30s timeout)
    result = await guarded.call_tool("run_sql", {"query": "SELECT * FROM users"})
    print(f"  run_sql (custom config): {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard + MCP Integration Demo")
    print("=" * 60)

    asyncio.run(demo_guarded_server())
    asyncio.run(demo_guarded_client())
    asyncio.run(demo_per_tool_config())

    print("\n✓ MCP example complete!")


if __name__ == "__main__":
    main()
