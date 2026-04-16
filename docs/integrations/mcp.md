# MCP Integration

## Overview

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard for connecting AI assistants to external data sources and tools. agentguard provides `GuardedMCPServer` (server-side) and `GuardedMCPClient` (client-side) wrappers.

## Installation

```bash
pip install awesome-agentguard mcp
```

---

## `GuardedMCPServer`

Wrap an MCP server to apply agentguard protection to all tool calls it receives:

```python
from agentguard.integrations import GuardedMCPServer
from agentguard import GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig

config = GuardConfig(
    validate_input=True,
    detect_hallucination=True,
    max_retries=2,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    record=True,
)

# Wrap your MCP server
guarded_server = GuardedMCPServer(original_server, config=config)
# guarded_server is a drop-in replacement
```

### Per-tool configuration

```python
guarded_server = GuardedMCPServer(
    original_server,
    config=GuardConfig(validate_input=True),          # Default config
    tool_configs={
        "database_query": GuardConfig(                  # Override for this tool
            validate_input=True,
            max_retries=3,
            circuit_breaker=CircuitBreakerConfig(failure_threshold=3),
        ),
        "send_email": GuardConfig(
            validate_input=True,
            rate_limit=RateLimitConfig(calls_per_hour=100),
        ),
    },
)
```

### Async and sync dispatch

```python
# Async (MCP standard)
result = await guarded_server.call_tool("search", {"query": "Python"})

# Sync
result = guarded_server.call_tool_sync("search", {"query": "Python"})
```

---

## `GuardedMCPClient`

Wrap an MCP client to guard outgoing tool calls before they're sent to a remote server:

```python
from agentguard.integrations import GuardedMCPClient
from agentguard import GuardConfig

client = GuardedMCPClient(
    original_client,
    config=GuardConfig(record=True, trace_backend="sqlite", trace_dir="./mcp_traces"),
)

# Calls are traced and timed
result = await client.call_tool("search", {"query": "Python"})

# All other client methods pass through transparently
tools = await client.list_tools()
```

---

## Full MCP Server Example

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from agentguard.integrations import GuardedMCPServer
from agentguard import GuardConfig

# Your MCP server
server = Server("my-tools")

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list:
    if name == "search_web":
        return [{"type": "text", "text": f"Results for {arguments['query']}"}]
    raise ValueError(f"Unknown tool: {name}")

# Wrap with agentguard
guarded = GuardedMCPServer(
    server,
    config=GuardConfig(validate_input=True, max_retries=2, record=True),
)

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await guarded.run(read_stream, write_stream, guarded.create_initialization_options())

asyncio.run(main())
```

---

## Transparent Proxying

Both `GuardedMCPServer` and `GuardedMCPClient` use `__getattr__` to proxy all non-intercepted methods to the underlying object:

```python
# These work transparently — not intercepted by agentguard
tools = await guarded_server.list_tools()
resources = await guarded_server.list_resources()
```

---

## Troubleshooting

### `mcp` package not found

Install it: `pip install mcp`. The `mcp` Python SDK is the reference implementation from Anthropic.

### Async tool dispatch hangs

Ensure your underlying MCP server's `call_tool` is properly async. If it's sync, wrap it:

```python
import asyncio

async def async_call_tool(name, args):
    return await asyncio.to_thread(server.call_tool_sync, name, args)
```
