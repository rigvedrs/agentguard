"""OpenAI-compatible provider example for agentguard.

Works with OpenRouter, Groq, Together AI, Fireworks AI, DeepInfra,
Mistral, xAI, and any other OpenAI-compatible API.

These providers all use the same OpenAI request/response format,
so agentguard's tool schemas and executor work with all of them.

Run with: python examples/openai_compatible.py
(Works in mock mode without any API keys.)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from agentguard import GuardConfig, guard
from agentguard.integrations import Provider, Providers, guard_tools


# ---------------------------------------------------------------------------
# Define tools (same tools work across ALL providers)
# ---------------------------------------------------------------------------


@guard(validate_input=True, max_retries=2)
def search_web(query: str) -> dict:
    """Search the web for information."""
    time.sleep(0.02)
    return {
        "query": query,
        "results": [
            {"title": f"Result 1 for '{query}'", "url": "https://example.com/1"},
            {"title": f"Result 2 for '{query}'", "url": "https://example.com/2"},
        ],
        "total": 2,
    }


@guard(validate_input=True, timeout=10.0)
def get_weather(city: str, units: str = "metric") -> dict:
    """Get current weather for a city."""
    time.sleep(0.01)
    return {
        "city": city,
        "temperature": 22.5,
        "conditions": "sunny",
        "units": units,
    }


@guard(validate_input=True)
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters"}
    result = eval(expression, {"__builtins__": {}})  # noqa: S307
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# Show available providers
# ---------------------------------------------------------------------------


def show_providers() -> None:
    """List all pre-built provider presets."""
    print("\n--- Supported OpenAI-Compatible Providers ---")
    for p in Providers.all():
        key_status = "✓ configured" if p.get_api_key() else "✗ not set"
        tools = "tool calling" if p.supports_tools else "no tool calling"
        print(f"  {p.name:<16} {p.base_url:<45} {p.env_key:<25} [{key_status}] ({tools})")


# ---------------------------------------------------------------------------
# Mock mode: demonstrate schema generation + execution
# ---------------------------------------------------------------------------


def demo_mock_mode() -> None:
    """Show that the same tools produce the same schemas for all providers."""
    print("\n--- Mock Mode (no API key required) ---")

    config = GuardConfig(validate_input=True, max_retries=1, record=True)
    executor = guard_tools([search_web, get_weather, calculate], config=config)

    print(f"\nGenerated {len(executor.tools)} tool schemas (OpenAI format):")
    for schema in executor.tools:
        fn = schema["function"]
        params = list(fn["parameters"]["properties"].keys())
        print(f"  {fn['name']}: {fn['description']}")
        print(f"    params: {params}")

    # Simulate tool calls (same format across all providers)
    class MockToolCall:
        def __init__(self, tc_id: str, name: str, args: str) -> None:
            self.id = tc_id
            class Fn:
                pass
            self.function = Fn()
            self.function.name = name  # type: ignore
            self.function.arguments = args  # type: ignore

    mock_calls = [
        MockToolCall("call_1", "search_web", '{"query": "agentguard python"}'),
        MockToolCall("call_2", "get_weather", '{"city": "Tokyo"}'),
        MockToolCall("call_3", "calculate", '{"expression": "2 ** 10"}'),
    ]

    print("\nExecuting mock tool calls:")
    results = executor.execute_all(mock_calls)
    for r in results:
        data = json.loads(r["content"])
        preview = json.dumps(data)[:80]
        print(f"  {r['tool_call_id']}: {preview}...")


# ---------------------------------------------------------------------------
# Live provider examples (only run if keys are set)
# ---------------------------------------------------------------------------


def demo_live_provider(provider: Provider, model: str) -> None:
    """Run a live tool-calling loop with a specific provider."""
    key = provider.get_api_key()
    if not key:
        print(f"\n  ({provider.env_key} not set; skipping {provider.name} live demo)")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("\n  (openai package not installed; skipping live demo)")
        return

    print(f"\n--- Live: {provider.name} ({model}) ---")

    config = GuardConfig(validate_input=True, max_retries=2)
    executor = guard_tools([search_web, get_weather, calculate], config=config)

    client = OpenAI(**provider.client_kwargs())

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in London and what is 42 * 7?"}],
        tools=executor.tools,
    )

    message = response.choices[0].message
    if message.tool_calls:
        print(f"  Model requested {len(message.tool_calls)} tool call(s)")
        results = executor.execute_all(message.tool_calls)
        for r in results:
            data = json.loads(r["content"])
            print(f"  → {json.dumps(data)[:80]}...")

        # Get final answer
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "What's the weather in London and what is 42 * 7?"},
            message.model_dump(),
        ]
        messages.extend(results)

        final = client.chat.completions.create(model=model, messages=messages)
        print(f"  Final: {final.choices[0].message.content}")
    else:
        print(f"  Response: {message.content}")


# ---------------------------------------------------------------------------
# Custom provider demo
# ---------------------------------------------------------------------------


def demo_custom_provider() -> None:
    """Show how to define a custom provider."""
    print("\n--- Custom Provider Definition ---")

    my_provider = Provider(
        name="My Custom LLM",
        base_url="https://api.my-llm-provider.com/v1",
        env_key="MY_LLM_API_KEY",
        default_model="my-model-v1",
        default_headers={"X-Custom-Header": "agentguard"},
    )
    print(f"  Provider: {my_provider}")
    print(f"  Client kwargs: { {k: '***' if k == 'api_key' else v for k, v in my_provider.client_kwargs(api_key='sk-test').items()} }")

    # Look up by name
    groq = Providers.by_name("groq")
    print(f"  Lookup 'groq': {groq}")

    openrouter = Providers.by_name("OpenRouter")
    print(f"  Lookup 'OpenRouter': {openrouter}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard — OpenAI-Compatible Provider Demo")
    print("=" * 60)
    print("\nagentguard works with ANY OpenAI-compatible provider.")
    print("Same tools, same schemas, same executor — just change the base_url.")

    show_providers()
    demo_mock_mode()
    demo_custom_provider()

    # Try live demos for any provider that has a key set
    live_providers = [
        (Providers.OPENROUTER, "openai/gpt-4o-mini"),
        (Providers.GROQ, "llama-3.3-70b-versatile"),
        (Providers.TOGETHER, "Qwen/Qwen2.5-7B-Instruct-Turbo"),
        (Providers.FIREWORKS, "accounts/fireworks/models/llama-v3p1-70b-instruct"),
        (Providers.DEEPINFRA, "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        (Providers.MISTRAL, "mistral-large-latest"),
    ]

    for provider, model in live_providers:
        demo_live_provider(provider, model)

    print("\n✓ OpenAI-compatible provider example complete!")


if __name__ == "__main__":
    main()
