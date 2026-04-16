# agentguard

**Runtime budget control and tool-call reliability for AI agents.**

[![PyPI version](https://img.shields.io/pypi/v/awesome-agentguard?color=blue)](https://pypi.org/project/awesome-agentguard/)
[![MIT License](https://img.shields.io/github/license/rigvedrs/agentguard)](https://github.com/rigvedrs/agentguard/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/github/actions/workflow/status/rigvedrs/agentguard/ci.yml?label=tests)](https://github.com/rigvedrs/agentguard/actions)

---

AI agents overspend, call tools with wrong parameters, and trust broken tool responses. `agentguard` is a lightweight Python runtime that keeps agent runs inside budget and makes tool calls trustworthy with spend caps, response verification, validation, retries, and tracing.

Works with **OpenAI**, **Anthropic**, **OpenRouter**, **LangChain**, **CrewAI**, **AutoGen**, **MCP**, or any Python function. Only core dependency: `pydantic`.

> New to agentguard? Start with the guided onboarding site: [agentguard-site](https://rigvedrs.github.io/agentguard-site/).

```python
from agentguard import guard

@guard(validate_input=True, verify_response=True, max_retries=3)
def search_web(query: str) -> dict:
    return requests.get(f"https://api.search.com?q={query}").json()
```

## Why agentguard?

agentguard is built around two jobs:

1. **Budget control** for agent runs, model calls, and shared multi-agent workflows.
2. **Tool-call reliability** so models can only act on tool results you actually trust.

| Problem | Without agentguard | With agentguard |
|---|---|---|
| **Cost spirals** | One prompt change causes 10x token costs | Per-call and per-session budget enforcement, real LLM spend tracking, and shared budget pools |
| **Malformed tool responses** | Schema drift, missing fields, anomalous values — no error raised | Multi-signal response verification with confidence scoring |
| **Invalid parameters** | Wrong types crash silently or produce garbage | Automatic validation from type hints + Pydantic schemas |
| **Cascading failures** | One failing tool takes down the entire agent | Circuit breakers with CLOSED → OPEN → HALF_OPEN state machine |
| **Rate limit violations** | Agent gets blocked mid-task | Token bucket rate limiting per-second, per-minute, per-hour |
| **No regression tests** | Agent breaks silently after refactoring | Auto-generate pytest tests from production traces |

## Quick Install

```bash
pip install awesome-agentguard
```

With integrations:

```bash
pip install awesome-agentguard[all]    # All integrations
pip install awesome-agentguard[costs]  # LiteLLM-backed real LLM cost tracking
pip install awesome-agentguard[rich]   # Rich terminal output
```

## Five-Minute Tour

### Put a hard cap on spend

```python
import os
from openai import OpenAI

from agentguard import TokenBudget
from agentguard.integrations import guard_openai_client

budget = TokenBudget(max_cost_per_session=5.00, max_calls_per_session=100)
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

### Guard tool execution

Apply `@guard` to any function and get tracing, timing, and error recording immediately:

```python
from agentguard import guard

@guard
def get_weather(city: str) -> dict:
    return {"temperature": 72, "city": city}

result = get_weather("NYC")
```

### Verify tool responses against expected contracts

```python
from agentguard import ResponseVerifier

verifier = ResponseVerifier(threshold=0.6)
verifier.register_tool(
    "get_weather",
    expected_latency_ms=(100, 5000),
    required_fields=["temperature", "humidity"],
)

result = verifier.verify(
    tool_name="get_weather",
    execution_time_ms=0.3,  # Sub-ms — no real I/O occurred
    response={"temperature": 72},  # Also missing "humidity"
)
print(result.is_anomalous)  # True
print(result.confidence)    # 0.95
```

### Production-ready configuration

```python
from agentguard import guard, CircuitBreaker, TokenBudget, RateLimiter

@guard(
    validate_input=True,
    validate_output=True,
    verify_response=True,       # checks timing, schema, patterns
    max_retries=3,
    timeout=30.0,
    budget=TokenBudget(max_cost_per_session=5.00, max_calls_per_session=100).config,
    circuit_breaker=CircuitBreaker(failure_threshold=5, recovery_timeout=60).config,
    rate_limit=RateLimiter(calls_per_minute=30).config,
    record=True,
)
def query_database(sql: str, limit: int = 100) -> list[dict]:
    return db.execute(sql, limit=limit)
```

### Auto-generate tests from production traces

```python
from agentguard import record_session
from agentguard.testing import TestGenerator

with record_session("./traces", backend="sqlite"):
    result = query_database("SELECT * FROM users LIMIT 10")

generator = TestGenerator(traces_dir="./traces")
generator.generate_tests(output="tests/test_generated.py")
```

## Framework Support

| Framework | Integration |
|---|---|
| OpenAI | `agentguard.integrations.OpenAIToolExecutor` |
| Anthropic | `agentguard.integrations.AnthropicToolExecutor` |
| LangChain | `agentguard.integrations.GuardedLangChainTool` |
| MCP | `agentguard.integrations.GuardedMCPServer` |
| CrewAI | `agentguard.integrations.GuardedCrewAITool` |
| AutoGen | `agentguard.integrations.GuardedAutoGenTool` |
| OpenRouter / Groq / Together | `agentguard.integrations.guard_tools` |
| Real LLM spend tracking | `guard_openai_client`, `guard_anthropic_client`, `guard_openai_compatible_client` |
| Any Python function | `@guard` decorator |

## Next Steps

- [Getting Started](getting-started.md) — install, first example, 5-minute tutorial
- [@guard Decorator](guides/guard-decorator.md) — every option explained with examples
- [Response Verification](guides/response-verification.md) — how anomaly detection works under the hood
- [API Reference](reference/api.md) — complete class and function reference
- Prefer a more visual introduction? Visit the onboarding site: [agentguard-site](https://rigvedrs.github.io/agentguard-site/)
