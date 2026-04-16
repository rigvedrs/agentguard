"""OpenAI integration example for agentguard.

Shows how to wrap tools for use with the OpenAI API, execute tool calls
from responses, and review traces.

Run with: python examples/openai_example.py
(Requires OPENAI_API_KEY environment variable — or run in mock mode without it.)
"""

from __future__ import annotations

import json
import os
import time

from agentguard import GuardConfig, guard
from agentguard.integrations import OpenAIToolExecutor, guard_openai_tools

# ---------------------------------------------------------------------------
# Define the tools
# ---------------------------------------------------------------------------


@guard(validate_input=True, max_retries=2, record=True, trace_dir="/tmp/agentguard_openai")
def search_web(query: str) -> dict:
    """Search the web for information about the given query."""
    time.sleep(0.02)  # Simulate network call
    return {
        "query": query,
        "results": [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2"},
        ],
        "total": 2,
    }


@guard(validate_input=True, timeout=10.0)
def get_weather(city: str, units: str = "metric") -> dict:
    """Get current weather for a city."""
    time.sleep(0.01)
    return {
        "city": city,
        "temperature": 18.5,
        "conditions": "partly cloudy",
        "units": units,
    }


@guard(validate_input=True)
def calculate(expression: str) -> dict:
    """Evaluate a simple mathematical expression."""
    try:
        # Safe evaluation of simple math expressions
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Expression contains forbidden characters"}
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


# ---------------------------------------------------------------------------
# Build the executor
# ---------------------------------------------------------------------------


def build_executor() -> OpenAIToolExecutor:
    config = GuardConfig(validate_input=True, max_retries=1, record=True)
    executor = OpenAIToolExecutor(config=config)
    executor.register(search_web)
    executor.register(get_weather)
    executor.register(calculate)
    return executor


# ---------------------------------------------------------------------------
# Mock mode (no API key required)
# ---------------------------------------------------------------------------


def run_mock_mode(executor: OpenAIToolExecutor) -> None:
    """Simulate the tool-calling loop without an actual OpenAI API call."""
    print("\n--- Mock Mode (no API key required) ---")
    print(f"Available tools: {[t['function']['name'] for t in executor.tools]}")

    # Simulate a tool call from OpenAI
    class MockToolCall:
        def __init__(self, tool_id: str, name: str, arguments: str) -> None:
            self.id = tool_id

            class Fn:
                pass

            self.function = Fn()
            self.function.name = name  # type: ignore
            self.function.arguments = arguments  # type: ignore

    mock_calls = [
        MockToolCall("call_1", "search_web", '{"query": "agentguard python library"}'),
        MockToolCall("call_2", "get_weather", '{"city": "London"}'),
        MockToolCall("call_3", "calculate", '{"expression": "42 * 3.14"}'),
    ]

    results = executor.execute_all(mock_calls)
    for r in results:
        data = json.loads(r["content"])
        print(f"  {r['tool_call_id']}: {json.dumps(data, indent=2)[:100]}...")


# ---------------------------------------------------------------------------
# Live OpenAI mode
# ---------------------------------------------------------------------------


def run_live_mode(executor: OpenAIToolExecutor) -> None:
    """Run a real conversation with tool calling."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": "What's the weather in Paris and what is 7 * 8?"}]

    print("\n--- Live OpenAI Tool Calling ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=executor.tools,
    )

    message = response.choices[0].message
    if message.tool_calls:
        print(f"  Model requested {len(message.tool_calls)} tool call(s)")
        tool_results = executor.execute_all(message.tool_calls)
        messages.append(message.model_dump())
        messages.extend(tool_results)

        # Final response
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        print(f"  Final response: {final.choices[0].message.content}")
    else:
        print(f"  No tool calls, response: {message.content}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard + OpenAI Integration Demo")
    print("=" * 60)

    executor = build_executor()
    run_mock_mode(executor)

    if os.getenv("OPENAI_API_KEY"):
        run_live_mode(executor)
    else:
        print("\n(Set OPENAI_API_KEY to run the live mode)")

    # Show tool schemas
    print("\n--- OpenAI Tool Schemas ---")
    for schema in executor.tools:
        fn = schema["function"]
        print(f"  {fn['name']}: {fn['description']}")
        print(f"    params: {list(fn['parameters']['properties'].keys())}")

    print("\n✓ OpenAI example complete!")


if __name__ == "__main__":
    main()
