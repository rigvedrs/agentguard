# AutoGen Integration

## Overview

[AutoGen](https://microsoft.github.io/autogen/) (Microsoft) is a multi-agent framework for building LLM-powered applications. Tools in AutoGen are registered with agent pairs using `register_for_llm` and `register_for_execution` decorators.

The agentguard AutoGen integration lets you apply protection to tool functions before they're registered — preserving AutoGen's introspection and JSON schema generation.

## Installation

```bash
pip install awesome-agentguard pyautogen
# or for AutoGen 0.4+
pip install awesome-agentguard autogen-agentchat
```

---

## Quick Start

```python
import autogen
import os
from agentguard.integrations import guard_autogen_tool, register_guarded_tools
from agentguard import GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig

# LLM configuration
llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}],
}

# Create AutoGen agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)

# Define your tools
def search_web(query: str) -> str:
    """Search the web for information."""
    import requests
    return requests.get(f"https://search.api.com?q={query}").text

def query_db(sql: str, limit: int = 100) -> list:
    """Query the database."""
    return db.execute(sql, limit=limit)

# Guard and register in one call
config = GuardConfig(
    validate_input=True,
    detect_hallucination=True,
    max_retries=2,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    rate_limit=RateLimitConfig(calls_per_minute=60),
    record=True,
)

guarded = register_guarded_tools(
    tools=[search_web, query_db],
    llm_agent=assistant,
    executor_agent=user_proxy,
    config=config,
)

# Start the conversation
user_proxy.initiate_chat(
    assistant,
    message="Research the latest developments in quantum computing",
)
```

---

## The `@guard_autogen_tool` Decorator

Apply directly to function definitions:

```python
from agentguard.integrations import guard_autogen_tool
from agentguard import GuardConfig

config = GuardConfig(validate_input=True, max_retries=2)

# Decorator with config
@guard_autogen_tool(config=config)
def search_web(query: str) -> str:
    """Search the web for information."""
    return requests.get(f"https://search.api.com?q={query}").text

# Then register with AutoGen as normal
@assistant.register_for_llm(description="Search the web for current information")
@user_proxy.register_for_execution()
def search_web_tool(query: str) -> str:
    return search_web(query=query)
```

Or use `as_function()` to get a signature-preserving proxy:

```python
@guard_autogen_tool(config=config)
def search_web(query: str) -> str:
    """Search the web."""
    return "results"

# Register the proxy directly
proxy = search_web.as_function()

@assistant.register_for_llm(description="Search the web")
@user_proxy.register_for_execution()
def search_web_tool(query: str) -> str:
    return proxy(query=query)
```

---

## The `register` Helper

`GuardedAutoGenTool.register` handles the registration boilerplate:

```python
from agentguard.integrations import GuardedAutoGenTool

def search_web(query: str) -> str:
    """Search the web."""
    return requests.get(f"...?q={query}").text

guarded = GuardedAutoGenTool(search_web, config=config)
guarded.register(
    llm_agent=assistant,
    executor_agent=user_proxy,
    description="Search the web for current information",
)
```

This is equivalent to:

```python
@assistant.register_for_llm(description="Search the web for current information")
@user_proxy.register_for_execution()
def search_web_proxy(query: str) -> str:
    return guarded(query=query)
```

---

## Bulk Registration with `guard_autogen_tools`

```python
from agentguard.integrations import guard_autogen_tools

guarded = guard_autogen_tools(
    [search_web, query_db, send_email],
    config=config,
)

# guarded is a dict: {"search_web": GuardedAutoGenTool, ...}
for tool in guarded.values():
    tool.register(assistant, user_proxy)

# Or use register_guarded_tools which does both in one call
```

---

## API Reference

### `guard_autogen_tool(func=None, *, config=None, description=None)`

Decorator to wrap an AutoGen tool with agentguard protection.

```python
# Without parentheses
@guard_autogen_tool
def my_tool(x: str) -> str: ...

# With config
@guard_autogen_tool(config=GuardConfig(max_retries=2))
def my_tool(x: str) -> str: ...
```

**Returns:** `GuardedAutoGenTool`

---

### `guard_autogen_tools(tools, config=None)`

Wrap a list of callables. Returns a `dict[str, GuardedAutoGenTool]`.

```python
guarded = guard_autogen_tools([search_web, query_db], config=config)
guarded["search_web"](query="Python")
```

---

### `register_guarded_tools(tools, llm_agent, executor_agent, config=None)`

Guard and register tools in one call. Returns `dict[str, GuardedAutoGenTool]`.

```python
guarded = register_guarded_tools(
    [search_web, query_db],
    llm_agent=assistant,
    executor_agent=user_proxy,
    config=config,
)
```

---

### `GuardedAutoGenTool`

| Attribute / Method | Description |
|---|---|
| `name` | Function name |
| `description` | First docstring line (or override) |
| `guarded_fn` | The underlying `GuardedTool` |
| `__wrapped__` | Original unwrapped function |
| `__call__(*args, **kwargs)` | Execute through agentguard |
| `acall(*args, **kwargs)` | Async execution |
| `as_function()` | Return signature-preserving proxy |
| `register(llm_agent, executor_agent, *, description=None)` | Register with an AutoGen agent pair |

---

## Preserving AutoGen Schema Generation

AutoGen uses `inspect.signature` to generate JSON schemas for the LLM. `GuardedAutoGenTool.as_function()` returns a `functools.wraps`-wrapped proxy that preserves the original signature:

```python
import inspect
from agentguard.integrations import GuardedAutoGenTool

def search_web(query: str, num_results: int = 10) -> str:
    """Search the web."""
    ...

guarded = GuardedAutoGenTool(search_web)
proxy = guarded.as_function()

# AutoGen will see the correct signature
print(inspect.signature(proxy))
# (query: str, num_results: int = 10) -> str
```

---

## Tracing AutoGen Agent Conversations

```python
from agentguard import GuardConfig
from agentguard.integrations import register_guarded_tools

config = GuardConfig(
    record=True,
    trace_backend="sqlite",
    trace_dir="./autogen_traces",
    session_id="conversation_001",
)

guarded = register_guarded_tools(
    [search_web, query_db],
    assistant, user_proxy,
    config=config,
)

user_proxy.initiate_chat(assistant, message="Research quantum computing")

# Analyse what happened
from agentguard.core.trace import TraceStore
store = TraceStore("./autogen_traces", backend="sqlite")
entries = store.read_session("conversation_001")

for entry in entries:
    print(f"{entry.tool_name}: {entry.result.execution_time_ms:.0f}ms")
```

---

## Multi-Agent Hierarchies

For AutoGen's nested chat and group chat patterns:

```python
import autogen
from agentguard.integrations import guard_autogen_tool, GuardedAutoGenTool
from agentguard import GuardConfig

config = GuardConfig(validate_input=True, max_retries=2, record=True)

# Wrap tools
@guard_autogen_tool(config=config)
def search_web(query: str) -> str:
    """Search the web."""
    ...

@guard_autogen_tool(config=config)
def analyze_data(data: str) -> str:
    """Analyze the provided data."""
    ...

# Create specialized agents
researcher = autogen.AssistantAgent("researcher", llm_config=llm_config)
analyst = autogen.AssistantAgent("analyst", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent("user_proxy", human_input_mode="NEVER")

# Register tools with appropriate agents
@researcher.register_for_llm(description="Search the internet")
@user_proxy.register_for_execution()
def search(query: str) -> str:
    return search_web(query=query)

@analyst.register_for_llm(description="Analyze data")
@user_proxy.register_for_execution()
def analyze(data: str) -> str:
    return analyze_data(data=data)
```

---

## Troubleshooting

### `ImportError: autogen is not installed`

Install it: `pip install pyautogen` (v0.2) or `pip install autogen-agentchat` (v0.4+).

### `AttributeError: 'AssistantAgent' has no attribute 'register_for_llm'`

This API exists in AutoGen 0.2+. Check your AutoGen version:

```bash
pip show pyautogen | grep Version
```

### JSON schema generation fails

AutoGen's schema generation requires type annotations. Ensure your tool functions have full annotations:

```python
# Good
def search(query: str, limit: int = 10) -> str: ...

# Bad — missing return annotation
def search(query: str, limit: int = 10): ...
```

### Guard not applied to all tool calls

Ensure you're calling the guarded version (or the proxy returned by `as_function()`), not the original function. Use `repr(tool)` to check: `GuardedAutoGenTool(name='search_web', ...)` vs just `<function search_web at 0x...>`.
