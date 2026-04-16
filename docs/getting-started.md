# Getting Started

This page gets you from zero to a budget-aware, tool-safe AI agent workflow in under five minutes.

## Installation

```bash
pip install awesome-agentguard
```

**Requirements:** Python 3.10+ · `pydantic >= 2.0` (only required dependency)

### Optional extras

```bash
pip install awesome-agentguard[all]     # OpenAI + Anthropic + LangChain integrations
pip install awesome-agentguard[costs]   # LiteLLM-backed real LLM cost tracking
pip install awesome-agentguard[rich]    # Colour terminal output via Rich
pip install awesome-agentguard[dev]     # Development tools (pytest, mypy, ruff)
```

Verify the installation:

```python
import agentguard
print(agentguard.__version__)
```

---

## Start With the Two Core Wins

The fastest way to understand `agentguard` is:

1. Put a hard cap on model spend.
2. Guard tool execution so bad tool calls and broken responses stop early.

## Step 1: Put a hard cap on spend

```python
import os
from openai import OpenAI

from agentguard import TokenBudget
from agentguard.integrations import guard_openai_client

budget = TokenBudget(
    max_cost_per_session=5.00,
    max_calls_per_session=100,
    alert_threshold=0.80,
)

client = guard_openai_client(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    budget=budget,
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this document"}],
)

print(budget.session_spend)
```

This is the quickest path to preventing runaway retries, overly expensive prompts, or model swaps from turning into a surprise bill.

## Your First Guarded Tool

### Step 2: Wrap a function with `@guard`

The simplest usage — zero configuration, all defaults:

```python
from agentguard import guard

@guard
def get_weather(city: str) -> dict:
    """Fetch current weather for a city."""
    import requests
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

# Every call is now traced, timed, and monitored
result = get_weather("San Francisco")
```

That's it. Every call now:

- Records the call with timing information
- Tracks success/failure counts in the global tool registry
- Propagates exceptions with full traceability

### Step 3: Add input validation

With type hints already on your function, validation is free:

```python
from agentguard import guard

@guard(validate_input=True)
def search_database(query: str, limit: int = 50) -> list[dict]:
    """Search the database."""
    if limit > 1000:
        raise ValueError("limit must be <= 1000")
    return db.search(query, limit=limit)
```

If an AI agent passes `limit="all"` instead of an integer, agentguard raises a `ValidationError` before the function even runs.

### Step 4: Add retries and timeout

```python
from agentguard import guard

@guard(
    validate_input=True,
    max_retries=3,       # Retry up to 3 times on failure
    timeout=10.0,        # Abort if it takes longer than 10 seconds
)
def call_external_api(endpoint: str) -> dict:
    """Call an external API endpoint."""
    import requests
    return requests.get(endpoint).json()
```

Retries use exponential backoff with jitter by default (1s, 2s, 4s + random jitter).

### Step 5: Add response verification

Verify that tool responses match expected contracts — correct timing, required fields, and expected patterns. Catches integration bugs, API schema drift, and test stubs accidentally left in production.

```python
from agentguard import guard

@guard(
    validate_input=True,
    verify_response=True,
    max_retries=2,
)
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price."""
    import requests
    return requests.get(f"https://finance-api.com/v1/price/{ticker}").json()

# Tell the verifier what normal responses look like
get_stock_price.register_response_profile(
    expected_latency_ms=(50, 3000),
    required_fields=["ticker", "price", "currency"],
)
```

When `verify_response=True`, agentguard checks:

1. **Execution time** — real API calls don't complete in < 2ms; sub-ms responses indicate no I/O occurred
2. **Required fields** — are all declared fields present in the response?
3. **Pattern matching** — does the response match expected regex patterns?
4. **Confidence scoring** — a weighted combination of all signals

### Step 6: Full production configuration

```python
from agentguard import guard, CircuitBreaker, TokenBudget, RateLimiter

@guard(
    validate_input=True,
    validate_output=True,
    verify_response=True,
    max_retries=3,
    timeout=30.0,
    budget=TokenBudget(
        max_cost_per_session=5.00,
        max_calls_per_session=100,
        alert_threshold=0.80,
    ).config,
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
    ).config,
    rate_limit=RateLimiter(
        calls_per_minute=30,
    ).config,
    record=True,
)
def query_database(sql: str, limit: int = 100) -> list[dict]:
    """Execute a SQL query and return results."""
    return db.execute(sql, limit=limit)
```

---

## Record and Generate Tests

In production, record real executions and turn them into a pytest test suite:

```python
from agentguard import record_session
from agentguard.testing import TestGenerator

# Record a session
with record_session("./traces", backend="sqlite"):
    result1 = get_weather("London")
    result2 = query_database("SELECT * FROM users LIMIT 5")

# Generate tests
generator = TestGenerator(traces_dir="./traces")
generator.generate_tests(output="tests/test_generated.py")
```

The generated file looks like:

```python
"""Auto-generated test suite from agentguard traces."""

def test_get_weather_0():
    """Recorded: get_weather('London')"""
    result = get_weather("London")
    assert isinstance(result, dict)

def test_query_database_0():
    """Recorded: query_database('SELECT * FROM users LIMIT 5')"""
    result = query_database("SELECT * FROM users LIMIT 5")
    assert isinstance(result, list)
```

---

## Using with an AI Framework

### OpenAI function calling

```python
import os
from openai import OpenAI
from agentguard import guard, GuardConfig
from agentguard.integrations import OpenAIToolExecutor

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@guard(validate_input=True, max_retries=2)
def get_weather(city: str) -> dict:
    """Get the weather for a city."""
    return {"city": city, "temperature": 72, "conditions": "sunny"}

@guard(validate_input=True)
def search_web(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"

# Register tools with agentguard's executor
executor = OpenAIToolExecutor()
executor.register(get_weather).register(search_web)

messages = [{"role": "user", "content": "What's the weather in Paris?"}]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=executor.tools,
)

# Execute tool calls through agentguard
results = executor.execute_all(response.choices[0].message.tool_calls)
```

### Real cost tracking for LLM calls

```python
import os
from openai import OpenAI

from agentguard import InMemoryCostLedger, TokenBudget
from agentguard.integrations import guard_openai_client

budget = TokenBudget(max_cost_per_session=5.00, max_calls_per_session=100)
budget.config.cost_ledger = InMemoryCostLedger()

client = guard_openai_client(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    budget=budget,
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise the latest report"}],
)

print(budget.session_spend)
```

This path reads provider-reported usage from the response, resolves pricing through LiteLLM when available, and falls back to an explicit `cost_per_call` only if you configured one.

### CrewAI

```python
from crewai import Agent, Task, Crew
from agentguard.integrations import guard_crewai_tools
from agentguard import GuardConfig

def search_web(query: str) -> str:
    """Search the web."""
    return f"results for {query}"

def analyze_data(data: str) -> str:
    """Analyze data."""
    return f"analysis of {data}"

config = GuardConfig(validate_input=True, max_retries=2, detect_hallucination=True)  # or verify_response=True
guarded_tools = guard_crewai_tools([search_web, analyze_data], config=config)

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You're an expert analyst.",
    tools=guarded_tools,
)
```

### AutoGen

```python
import autogen
import os
from agentguard.integrations import guard_autogen_tool
from agentguard import GuardConfig

config = GuardConfig(validate_input=True, max_retries=2)

@guard_autogen_tool(config=config)
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"results for {query}"

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}],
}
assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
user_proxy = autogen.UserProxyAgent("user_proxy", human_input_mode="NEVER")

@assistant.register_for_llm(description="Search the web")
@user_proxy.register_for_execution()
def search_web_tool(query: str) -> str:
    return search_web(query=query)
```

---

## What's Next?

- [@guard Decorator Reference](guides/guard-decorator.md) — every parameter explained
- [Response Verification](guides/response-verification.md) — how anomaly detection works
- [Circuit Breaker](guides/circuit-breaker.md) — prevent cascading failures
- [Testing Guide](guides/testing.md) — full trace-to-test workflow
- [API Reference](reference/api.md) — complete class documentation
