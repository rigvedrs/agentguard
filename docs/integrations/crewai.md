# CrewAI Integration

## Overview

[CrewAI](https://crewai.com) is a popular multi-agent framework that lets you build teams of AI agents collaborating to complete complex tasks. The agentguard CrewAI integration wraps your CrewAI tools with validation, hallucination detection, circuit breakers, and tracing — without changing how you define your agents or crews.

## Installation

```bash
pip install awesome-agentguard crewai
```

---

## Quick Start

```python
from crewai import Agent, Task, Crew
from crewai.tools import tool
from agentguard.integrations import guard_crewai_tools, GuardedCrewAITool
from agentguard import GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig

# Define your CrewAI tools as usual
@tool("Search the Web")
def search_web(query: str) -> str:
    """Search the internet for current information."""
    import requests
    return requests.get(f"https://search.api.com?q={query}").text

@tool("Query Database")
def query_db(sql: str) -> str:
    """Execute a read-only SQL query."""
    return db.execute(sql)

# Apply agentguard protection
config = GuardConfig(
    validate_input=True,
    detect_hallucination=True,
    max_retries=2,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    rate_limit=RateLimitConfig(calls_per_minute=60),
    record=True,
)

guarded_tools = guard_crewai_tools([search_web, query_db], config=config)

# Build your crew as normal — drop in the guarded tools
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI and provide detailed analysis",
    backstory="You are an expert analyst at a leading think tank.",
    tools=guarded_tools,  # ← guarded tools, same interface
    verbose=True,
)
```

---

## Supported Tool Styles

### `@tool` decorated functions

```python
from crewai.tools import tool

@tool("Search Web")
def search_web(query: str) -> str:
    """Search the web for current information."""
    return requests.get(f"...?q={query}").text

guarded = GuardedCrewAITool(search_web, config=config)
```

### `BaseTool` subclasses

```python
from crewai.tools import BaseTool
from agentguard.integrations import GuardedCrewAITool

class SearchTool(BaseTool):
    name: str = "Search Web"
    description: str = "Searches the internet."

    def _run(self, query: str) -> str:
        return requests.get(f"...?q={query}").text

search = SearchTool()
guarded = GuardedCrewAITool(search, config=config)
```

### Plain Python functions

Any callable can be wrapped:

```python
def search_web(query: str) -> str:
    """Search the web."""
    return f"results for {query}"

guarded = GuardedCrewAITool(search_web, config=config)
```

---

## API Reference

### `guard_crewai_tools(tools, config=None)`

Wrap a list of CrewAI tools in bulk. Recommended for protecting an entire toolset at once.

```python
from agentguard.integrations import guard_crewai_tools
from agentguard import GuardConfig

guarded = guard_crewai_tools(
    [search_web, query_db, send_email],
    config=GuardConfig(validate_input=True, max_retries=2),
)
```

**Parameters:**
- `tools` — list of `@tool` functions, `BaseTool` instances, or plain callables
- `config` — `GuardConfig` applied to all tools. Optional, defaults to zero-config.

**Returns:** `list[GuardedCrewAITool]`

---

### `GuardedCrewAITool(tool, config=None, *, name=None, description=None)`

Wrap a single CrewAI tool.

```python
from agentguard.integrations import GuardedCrewAITool

guarded = GuardedCrewAITool(
    search_web,
    config=GuardConfig(validate_input=True),
    name="Web Search",           # Override name
    description="Searches web",  # Override description
)
```

**Methods:**

| Method | Description |
|---|---|
| `run(*args, **kwargs)` | Execute through agentguard |
| `arun(*args, **kwargs)` | Async execution |
| `__call__(*args, **kwargs)` | Alias for `run()` |
| `to_crewai_tool()` | Return a native `BaseTool` subclass |
| `_run(*args, **kwargs)` | CrewAI `BaseTool._run` compatibility |
| `_arun(*args, **kwargs)` | CrewAI async compatibility |

**Attributes:**

| Attribute | Description |
|---|---|
| `name` | Tool name (from original or overridden) |
| `description` | Tool description |
| `guarded_fn` | The underlying `GuardedTool` |

---

## Converting to Native CrewAI Tools

If you need to pass tools to CrewAI APIs that require a proper `BaseTool` instance:

```python
guarded = GuardedCrewAITool(search_web, config=config)
crewai_tool = guarded.to_crewai_tool()

# crewai_tool is a proper BaseTool subclass
researcher = Agent(tools=[crewai_tool])
```

---

## Multi-Agent with Shared Configuration

When multiple agents share tools, apply guards once:

```python
from crewai import Agent, Task, Crew
from agentguard.integrations import guard_crewai_tools
from agentguard import GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig, BudgetConfig

# Shared guardrail config
shared_config = GuardConfig(
    validate_input=True,
    detect_hallucination=True,
    max_retries=2,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    rate_limit=RateLimitConfig(calls_per_minute=120),
    budget=BudgetConfig(max_cost_per_session=10.00),
    record=True,
    trace_dir="./agent_traces",
)

guarded_tools = guard_crewai_tools(
    [search_web, query_db, code_executor, document_reader],
    config=shared_config,
)

researcher = Agent(
    role="Researcher",
    goal="Research topics thoroughly",
    tools=guarded_tools[:2],   # search_web, query_db
)

analyst = Agent(
    role="Analyst",
    goal="Analyse research and produce reports",
    tools=guarded_tools[2:],   # code_executor, document_reader
)

crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    verbose=True,
)

result = crew.kickoff()
```

---

## Handling Errors in CrewAI

CrewAI agents handle tool errors by observing them in the agent's context. agentguard errors are standard Python exceptions, so CrewAI will see them like any other tool failure:

```python
from agentguard.core.types import (
    CircuitOpenError, BudgetExceededError, RateLimitError, ValidationError
)

# CrewAI sees these as tool errors and may retry or escalate
# agentguard provides rich error messages the LLM can act on
```

To add custom error handling, use an `after_call` hook:

```python
import logging

logger = logging.getLogger(__name__)

def log_tool_error(call, result):
    if result.failed:
        logger.error(
            f"Tool {call.tool_name} failed: {result.exception}",
            extra={"session_id": call.session_id, "tool": call.tool_name},
        )

config = GuardConfig(
    validate_input=True,
    after_call=log_tool_error,
)
```

---

## Tracing CrewAI Agent Runs

With `record=True`, every tool call in your crew is recorded to disk:

```python
config = GuardConfig(
    record=True,
    trace_backend="sqlite",
    trace_dir="./crew_traces",
    session_id="crew_run_001",
)

guarded_tools = guard_crewai_tools(tools, config=config)
crew = Crew(agents=[researcher, analyst], tasks=tasks)
result = crew.kickoff()

# Inspect traces
from agentguard.core.trace import TraceStore
store = TraceStore("./crew_traces", backend="sqlite")
entries = store.read_session("crew_run_001")

for entry in entries:
    print(f"{entry.tool_name}: {entry.result.status.value} in {entry.result.execution_time_ms:.0f}ms")
```

---

## Troubleshooting

### `ImportError: crewai is not installed`

Install it: `pip install crewai`. If you're calling `to_crewai_tool()`, crewai must be installed.

### `TypeError: Cannot extract a callable from ...`

The tool object doesn't match any supported pattern. Ensure it's a:
- `@crewai.tools.tool` decorated function
- `BaseTool` subclass instance with a `_run` method
- Plain Python callable

### Tools not being guarded

Verify you're passing the `guarded_tools` list (not the original tools) to the `Agent`. Check with `isinstance(tool, GuardedCrewAITool)`.

### Rate limits applied globally across agents

Rate limits are shared by effective key, not just by `GuardedCrewAITool`
instance. By default, tools with the same name share one bucket across agents.
Set `shared_key=""` in `RateLimitConfig` to force per-instance limits, or use
a non-empty `shared_key` to share a provider quota across different tool
names. If multiple tools register the same effective key with different rate
limits, the first config wins and agentguard emits a warning.
