# OpenAI Integration

## Overview

agentguard provides a first-class integration for OpenAI function calling. `OpenAIToolExecutor` handles tool registration, JSON schema generation, and guarded execution, while `guard_openai_client` adds response-based spend tracking and budget enforcement for the model side of the loop.

## Installation

```bash
pip install awesome-agentguard openai
pip install awesome-agentguard[costs] openai   # Optional: real LLM cost tracking
```

---

## Quick Start

```python
import os
from openai import OpenAI
from agentguard import guard, GuardConfig
from agentguard.integrations import OpenAIToolExecutor

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@guard(validate_input=True, max_retries=2, verify_response=True)
def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    import requests
    return requests.get(f"https://wttr.in/{city}?format=j1").json()

@guard(validate_input=True)
def search_web(query: str) -> str:
    """Search the web for current information."""
    return f"Search results for: {query}"

# Build the executor
executor = OpenAIToolExecutor()
executor.register(get_weather).register(search_web)

# Standard OpenAI agent loop
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=executor.tools,       # ← auto-generated JSON schemas
        tool_choice="auto",
    )
    
    message = response.choices[0].message
    
    if not message.tool_calls:
        print(message.content)
        break
    
    # Execute all tool calls with guards
    results = executor.execute_all(message.tool_calls)
    
    messages.append(message.to_dict())
    for call, result in zip(message.tool_calls, results):
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": str(result),
        })
```

---

## `OpenAIToolExecutor` API

### `register(func, config=None)`

Register a tool and return `self` for chaining:

```python
executor = OpenAIToolExecutor()
executor \
    .register(get_weather) \
    .register(search_web, config=GuardConfig(max_retries=1)) \
    .register(query_db)
```

### `executor.tools`

Returns the list of tool definitions in OpenAI's JSON schema format:

```python
tools = executor.tools
# [
#   {
#     "type": "function",
#     "function": {
#       "name": "get_weather",
#       "description": "Get the current weather for a city.",
#       "parameters": {
#         "type": "object",
#         "properties": {"city": {"type": "string"}},
#         "required": ["city"]
#       }
#     }
#   },
#   ...
# ]
```

### `execute_all(tool_calls)`

Execute a list of `ChatCompletionMessageToolCall` objects:

```python
results = executor.execute_all(response.choices[0].message.tool_calls)
# results[i] is the return value of the i-th tool call
```

---

## Real Cost Tracking for OpenAI Calls

Use `guard_openai_client` when you also want response-based cost tracking for model calls. This is one of the clearest `agentguard` production stories: model spend stays inside budget, and tool calls are validated and verified before the agent trusts them. The wrapped client preserves the normal OpenAI SDK interface, records usage from the response, prices it through LiteLLM when available, and updates your budget/traces/telemetry automatically.

```python
import os
from openai import OpenAI

from agentguard import InMemoryCostLedger, TokenBudget
from agentguard.integrations import OpenAIToolExecutor, guard_openai_client

budget = TokenBudget(max_cost_per_session=5.00, max_calls_per_session=100)
budget.config.model_pricing_overrides = {
    "my-private-model": (2.0, 6.0),  # $/1M input, $/1M output
}
budget.config.cost_ledger = InMemoryCostLedger()

client = guard_openai_client(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    budget=budget,
)

executor = OpenAIToolExecutor().register(get_weather).register(search_web)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=executor.tools,
)

results = executor.execute_all(response.choices[0].message.tool_calls)
print(budget.session_spend)
```

Notes:

- Pricing resolution order is override → LiteLLM → explicit `cost_per_call` fallback.
- If usage is present but pricing is unknown, agentguard records the usage and marks cost as unknown instead of inventing a price.
- Streaming responses are recorded once from the terminal usage-bearing event.

---

## Standalone Helpers

### `function_to_openai_tool(func)`

Convert a function to an OpenAI tool definition without the executor:

```python
from agentguard.integrations import function_to_openai_tool

schema = function_to_openai_tool(get_weather)
```

### `guard_openai_tools(functions, config=None)`

Wrap a list of functions and return an `OpenAIToolExecutor`:

```python
from agentguard.integrations import guard_openai_tools

executor = guard_openai_tools([get_weather, search_web], config=GuardConfig(max_retries=2))
```

### `execute_openai_tool_call(tool_call, tools, registry=None)`

Execute a single tool call:

```python
from agentguard.integrations import execute_openai_tool_call

result = execute_openai_tool_call(
    tool_call=response.choices[0].message.tool_calls[0],
    tools=executor.tools,
    registry={"get_weather": get_weather, "search_web": search_web},
)
```

---

## Async Agent Loops

For async applications with `openai`'s async client:

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    
    while True:
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=executor.tools,
        )
        
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content
        
        # Execute tools without blocking the event loop
        tasks = [
            asyncio.to_thread(executor.execute, tc) for tc in message.tool_calls
        ]
        results = await asyncio.gather(*tasks)
        
        messages.append(message.to_dict())
        for tc, result in zip(message.tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })
```

---

## Error Handling in the Agent Loop

agentguard errors are returned as tool result strings so the agent can see and react to them:

```python
from agentguard.core.types import CircuitOpenError, BudgetExceededError

def safe_execute(executor, tool_calls):
    results = []
    for tc in tool_calls:
        try:
            result = executor.execute(tc)
            results.append(str(result))
        except CircuitOpenError as e:
            results.append(f"Service temporarily unavailable. Retry in {e.recovery_in:.0f}s.")
        except BudgetExceededError:
            results.append("Budget limit reached. Please reduce scope.")
    return results
```

---

## Troubleshooting

### Schema generation fails for complex types

agentguard generates schemas from Python type hints. Pydantic models and `TypedDict` are supported. For complex nested types, use `Annotated` with a `description`:

```python
from typing import Annotated
from agentguard import guard

@guard(validate_input=True)
def search(
    query: Annotated[str, "The search query"],
    limit: Annotated[int, "Max number of results (1-100)"] = 10,
) -> list[dict]:
    ...
```

### `ValidationError` on tool call from GPT

GPT models sometimes pass integers as strings. Set `validate_input=True` to catch these early with a clear error message that the model can fix.
