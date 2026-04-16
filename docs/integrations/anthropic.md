# Anthropic Integration

## Overview

agentguard integrates with Anthropic's Claude models via the tool use API. `AnthropicToolExecutor` handles schema generation and guarded execution for Claude's native tool calling format.

## Installation

```bash
pip install awesome-agentguard anthropic
pip install awesome-agentguard[costs] anthropic   # Optional: real LLM cost tracking
```

---

## Quick Start

```python
import os
import anthropic
from agentguard import guard, GuardConfig
from agentguard.integrations import AnthropicToolExecutor, guard_anthropic_tools

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@guard(validate_input=True, max_retries=2, detect_hallucination=True)
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    import requests
    return requests.get(f"https://wttr.in/{city}?format=j1").json()

@guard(validate_input=True)
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Create tool definitions and executor
tools = guard_anthropic_tools([get_weather, search_web])
executor = AnthropicToolExecutor({"get_weather": get_weather, "search_web": search_web})

messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

# Agent loop
while True:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        tools=tools,
        messages=messages,
    )
    
    if response.stop_reason == "end_turn":
        # Extract text from content blocks
        text = next((b.text for b in response.content if b.type == "text"), "")
        print(text)
        break
    
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = executor.execute(block)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
        
        messages.append({"role": "user", "content": tool_results})
```

---

## `AnthropicToolExecutor` API

### `execute(tool_use_block)`

Execute a single `ToolUseBlock`:

```python
for block in response.content:
    if block.type == "tool_use":
        result = executor.execute(block)
```

### `execute_all(content_blocks)`

Execute all tool use blocks in a response:

```python
results = executor.execute_all(response.content)
```

---

## `guard_anthropic_tools(functions, config=None)`

Convert a list of guarded functions to Anthropic tool definitions:

```python
from agentguard.integrations import guard_anthropic_tools

tools_schema = guard_anthropic_tools([get_weather, search_web])
# Returns Anthropic-format tool definitions
```

---

## `function_to_anthropic_tool(func)`

Convert a single function to Anthropic's tool definition format:

```python
from agentguard.integrations import function_to_anthropic_tool

schema = function_to_anthropic_tool(get_weather)
# {"name": "get_weather", "description": "...", "input_schema": {...}}
```

---

## Extended Thinking with Tools

Claude 3.7 Sonnet and later models support extended thinking. agentguard works seamlessly:

```python
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=tools,
    messages=messages,
)

# Handle thinking blocks + tool use blocks
for block in response.content:
    if block.type == "thinking":
        pass  # Skip thinking blocks
    elif block.type == "tool_use":
        result = executor.execute(block)
```

---

## Real Cost Tracking for Anthropic Calls

Wrap the Anthropic client with `guard_anthropic_client` to record spend from Claude response usage automatically.

```python
import os
import anthropic

from agentguard import InMemoryCostLedger, TokenBudget
from agentguard.integrations import AnthropicToolExecutor, guard_anthropic_client

budget = TokenBudget(max_cost_per_session=5.00)
budget.config.cost_ledger = InMemoryCostLedger()

client = guard_anthropic_client(
    anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
    budget=budget,
)

executor = AnthropicToolExecutor({"get_weather": get_weather, "search_web": search_web})

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    tools=tools,
    messages=messages,
)

print(budget.session_spend)
```

The wrapper supports normal requests and streaming. Usage is extracted from Anthropic's response metadata, spend is recorded once per model call, and unknown pricing is surfaced honestly rather than replaced with a guessed amount.

---

## Troubleshooting

### `anthropic.BadRequestError` on tool call

Anthropic validates tool schemas strictly. Ensure all function parameters have type annotations and the return type is specified.

### Tool result not shown to model

Ensure tool results are wrapped in the correct format:

```python
{"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
```

The `content` field must be a string or a list of content blocks.
