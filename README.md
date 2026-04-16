<p align="center">
  <h1 align="center">🛡️ agentguard</h1>
  <p align="center"><b>Runtime budget control and tool-call reliability for AI agents</b></p>
  <p align="center">
    <a href="https://pypi.org/project/awesome-agentguard/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/awesome-agentguard?color=blue"></a>
    <a href="https://github.com/rigvedrs/agentguard/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/rigvedrs/agentguard"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
    <a href="https://github.com/rigvedrs/agentguard/actions"><img alt="CI Tests" src="https://img.shields.io/github/actions/workflow/status/rigvedrs/agentguard/ci.yml?label=tests"></a>
    <a href="https://pypi.org/project/awesome-agentguard/"><img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/awesome-agentguard"></a>
  </p>
</p>

---

**AI agents overspend, call tools with wrong parameters, and trust broken tool responses.** `agentguard` is a lightweight Python runtime that keeps agent runs inside budget and makes tool calls trustworthy with spend caps, response verification, validation, retries, and tracing.

Works with **OpenAI**, **Anthropic**, **OpenRouter**, **Groq**, **Together AI**, **Fireworks AI**, **LangChain**, **MCP**, or any Python function. Only dependency: `pydantic`.

```python
from agentguard import guard

@guard(validate_input=True, verify_response=True, max_retries=3)
def search_web(query: str) -> dict:
    return requests.get(f"https://api.search.com?q={query}").json()
```

## What agentguard is for

`agentguard` has two core jobs:

1. **Keep agent runs inside budget** with per-call, per-session, and shared multi-agent spend controls.
2. **Make tool calls trustworthy** with input validation, output validation, response verification, and execution safeguards.

Everything else in the library supports those two outcomes: retries, circuit breakers, rate limits, tracing, telemetry, benchmarking, and generated tests.

## Why teams reach for agentguard

| Problem | How AI agents fail today | How agentguard fixes it |
|---|---|---|
| **Cost spirals & runaway spending** | One prompt change, retry loop, or model escalation causes a surprise bill | Per-call and per-session budget enforcement, real usage-based LLM spend tracking, and shared multi-agent budget pools |
| **Malformed tool responses** | Tool returns missing fields, schema drift, or anomalous values — no error raised | Multi-signal response verification (timing, schema, patterns, statistical anomalies) |
| **Invalid tool parameters** | Agent passes wrong types or missing fields | Automatic input/output validation from Python type hints + Pydantic schemas |
| **Cascading failures** | One failing tool takes down the entire agent | Circuit breakers with CLOSED → OPEN → HALF_OPEN state machine |
| **API rate limit violations** | Agent exceeds rate limits, gets blocked | Token bucket rate limiting (per-second, per-minute, per-hour) |
| **No regression tests** | 40% of agent projects fail with no test suite | Auto-generate pytest tests from production traces |
| **Framework lock-in** | Each LLM framework has its own observability | Framework-agnostic — works with OpenAI, Anthropic, LangChain, MCP, or raw functions |

### The Problem in Numbers

- **82.6%** of Stack Overflow questions about AI agents have no accepted answer ([arXiv](https://arxiv.org/html/2510.25423v2))
- **40-95%** of agent projects fail between prototype and production
- **0** widely-adopted open-source libraries focused on runtime tool response verification in Python agents

## Install agentguard

```bash
pip install awesome-agentguard
```

With optional integrations:

```bash
pip install awesome-agentguard[all]        # OpenAI + Anthropic + LangChain integrations
pip install awesome-agentguard[costs]      # LiteLLM-backed real LLM cost tracking
pip install awesome-agentguard[rich]       # Colour terminal output
pip install awesome-agentguard[dashboard]  # Local trace dashboard extras
```

**Requirements:** Python 3.10+ · Only dependency: `pydantic>=2.0`

## Quick Start

### 1. Put a hard cap on agent spend

Use `TokenBudget` when you want the run to stop before a retry loop or model change burns money:

```python
import os
from agentguard import TokenBudget
from agentguard.integrations import guard_openai_client
from openai import OpenAI

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

### 2. Guard tool calls with one decorator

```python
from agentguard import guard

@guard
def get_weather(city: str) -> dict:
    """Every call is now traced, timed, and validated."""
    return {"temperature": 72, "city": city}

result = get_weather("NYC")
```

### 3. Verify tool responses against expected contracts

Detect when a tool response violates what you've defined as normal — anomalous execution timing, missing required fields, pattern mismatches, or statistically unusual values. Useful for catching schema drift, API contract changes, integration bugs, and misconfigured mocks.

```python
from agentguard import ResponseVerifier

verifier = ResponseVerifier(threshold=0.6)

# Register what normal responses look like for this tool
verifier.register_tool(
    "get_weather",
    expected_latency_ms=(100, 5000),        # Real API: 100ms–5s
    required_fields=["temperature", "humidity"],
    response_patterns=[r'"temperature":\s*-?\d+'],
)

# Check a response that came back suspiciously fast and incomplete
result = verifier.verify(
    tool_name="get_weather",
    execution_time_ms=0.3,                  # 0.3ms — no network call happened
    response={"temperature": 72, "conditions": "sunny"},
)

print(result.is_anomalous)   # True  (missing "humidity", sub-ms timing)
print(result.confidence)     # 0.95
print(result.reason)         # "Execution time 0.30ms is below the 2ms minimum for real I/O..."
```

### 4. Production-ready protection for tool execution

```python
from agentguard import guard, CircuitBreaker, TokenBudget, RateLimiter

@guard(
    validate_input=True,
    validate_output=True,
    verify_response=True,       # checks timing, schema, patterns
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
    record=True,  # Save traces for test generation
)
def query_database(sql: str, limit: int = 100) -> list[dict]:
    return db.execute(sql, limit=limit)
```

### 5. Auto-generate pytest tests from production traces

Record real agent executions, then auto-generate a pytest test suite for regression testing:

```python
from agentguard import TraceRecorder, record_session
from agentguard.testing import TestGenerator

# Record during production
with record_session("./traces", backend="sqlite"):
    result = query_database("SELECT * FROM users LIMIT 10")
    result = get_weather("San Francisco")

# Generate test file
generator = TestGenerator(traces_dir="./traces")
generator.generate_tests(output="tests/test_generated.py")
```

By default, SQLite-backed recording writes to `./traces/agentguard_traces.db`. Use `trace_backend="jsonl"` when you need legacy file-per-session traces.

Generated test file:

```python
"""Auto-generated test suite from agentguard production traces."""

def test_query_database_0():
    """Recorded: query_database('SELECT * FROM users LIMIT 10', limit=100)"""
    result = query_database("SELECT * FROM users LIMIT 10", limit=100)
    assert isinstance(result, list)

def test_get_weather_0():
    """Recorded: get_weather('San Francisco')"""
    result = get_weather("San Francisco")
    assert isinstance(result, dict)
    assert "temperature" in result
```

### 6. Fluent test assertions for agent tool calls

```python
from agentguard import assert_tool_call

# Build assertions on recorded trace entries
assert_tool_call(entry).succeeded().within_ms(5000).returned_dict().has_keys("temperature", "humidity")
```

### 7. Replay and diff agent traces

```python
from agentguard.testing import TraceReplayer

replayer = TraceReplayer(traces_dir="./traces")
results = replayer.replay_all(tools={"get_weather": get_weather})

for r in results:
    print(f"{r['tool_name']}: {'PASS' if r['match'] else 'FAIL'}")
```

## LLM Framework Integrations

### Any OpenAI-Compatible Provider (OpenRouter, Groq, Together, Fireworks, etc.)

agentguard works with **any OpenAI-compatible API** out of the box. One integration covers 10+ providers:

```python
from openai import OpenAI
from agentguard.integrations import guard_tools, Providers

# Same tools work across ALL providers — just change the provider
executor = guard_tools([search_web, get_weather])

# OpenRouter (300+ models)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# Groq (ultra-low latency)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

# Together AI, Fireworks, DeepInfra, Mistral, xAI — same pattern
client = OpenAI(**Providers.TOGETHER.client_kwargs())

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    tools=executor.tools,
    messages=[{"role": "user", "content": "Search for Python tutorials"}],
)
results = executor.execute_all(response.choices[0].message.tool_calls)
```

**Built-in provider presets:** OpenAI, OpenRouter, Groq, Together AI, Fireworks AI, DeepInfra, Mistral, Perplexity, Novita AI, xAI — or define your own with `Provider(name=..., base_url=..., env_key=...)`.

### OpenAI Function Calling with Guardrails

```python
from agentguard.integrations import guard_openai_tools, OpenAIToolExecutor

# Wrap your tools for OpenAI function calling
executor = OpenAIToolExecutor()
executor.register(search_web).register(get_weather)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=executor.tools,
)

# Execute all tool calls with guards
results = executor.execute_all(response.choices[0].message.tool_calls)
```

### Real LLM Cost Tracking

Wrap supported provider clients to record real token usage and pricing directly from API responses:

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
    messages=[{"role": "user", "content": "Summarise this page"}],
)

print(budget.session_spend)
```

Pricing resolution order:

1. `model_pricing_overrides`
2. LiteLLM pricing data
3. explicit `cost_per_call`
4. otherwise usage is tracked and cost is marked unknown

### Anthropic Claude Tool Use with Guardrails

```python
from agentguard.integrations import guard_anthropic_tools, AnthropicToolExecutor

tools = guard_anthropic_tools([search_web, get_weather])
executor = AnthropicToolExecutor({"search_web": search_web, "get_weather": get_weather})
```

### LangChain Agent Tool Validation

```python
from agentguard.integrations import GuardedLangChainTool, guard_langchain_tools

# Wrap existing LangChain tools
guarded = guard_langchain_tools([my_search_tool, my_db_tool])
```

### MCP (Model Context Protocol) Server Guards

```python
from agentguard.integrations import GuardedMCPServer

# Wrap an MCP server with guards
guarded_server = GuardedMCPServer(original_server, guards={
    "search": {"validate_input": True, "max_retries": 2},
    "database_query": {"budget": budget_config, "circuit_breaker": cb_config},
})
```

## Architecture — How agentguard Protects AI Agent Tool Calls

```
┌──────────────────────────────────────────────────────────┐
│                     Your AI Agent                         │
│              (OpenAI / Anthropic / LangChain / etc.)      │
└──────────────────────┬───────────────────────────────────┘
                       │ tool call
                       ▼
┌──────────────────────────────────────────────────────────┐
│                    @guard decorator                       │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Circuit    │  │    Rate     │  │   Budget    │     │
│  │   Breaker    │  │   Limiter   │  │  Enforcer   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                │                │              │
│         ▼                ▼                ▼              │
│  ┌─────────────────────────────────────────────┐        │
│  │           Input Validation                   │        │
│  │      (type hints + Pydantic schemas)         │        │
│  └─────────────────────┬───────────────────────┘        │
│                        │                                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │      Execute with Retry + Timeout            │        │
│  │      (exponential backoff, jitter)           │        │
│  └─────────────────────┬───────────────────────┘        │
│                        │                                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │       Response Verification                  │        │
│  │  (timing, schema, patterns, anomaly score)   │        │
│  └─────────────────────┬───────────────────────┘        │
│                        │                                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │        Output Validation                     │        │
│  └─────────────────────┬───────────────────────┘        │
│                        │                                 │
│                        ▼                                 │
│  ┌─────────────────────────────────────────────┐        │
│  │    Trace Recording → Test Generation         │        │
│  └─────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
              Your actual tool
```

## CLI — Inspect Agent Traces and Generate Tests

```bash
# Initialize a SQLite trace store
agentguard traces init ./traces

# List recorded agent traces
agentguard traces list ./traces

# Show trace details for a session
agentguard traces show agent_run_001 ./traces

# Get latency and failure statistics
agentguard traces stats ./traces

# Generate JSON report
agentguard traces report ./traces --output report.json

# Import legacy JSONL traces into SQLite
agentguard traces import ./legacy_traces ./traces

# Export traces for replay or offline analysis
agentguard traces export ./traces --output-dir ./trace-export

# Run the local dashboard
agentguard traces serve ./traces --port 8765

# Auto-generate pytest test suite from traces
agentguard generate ./traces --output tests/test_generated.py
```

## Full API Reference

### Core — Guard Decorator and Tool Registry

| Component | Description |
|---|---|
| `@guard` | Decorator that wraps any Python function with the full protection stack |
| `GuardConfig` | Configuration dataclass for all guard options |
| `GuardedTool` | The wrapper class created by `@guard` |
| `ToolRegistry` | Global registry for tool discovery, stats, and health checks |

### Validators — Response Verification and Schema Validation

| Component | Description |
|---|---|
| `ResponseVerifier` | Multi-signal response anomaly detection: timing, schema, patterns, statistical values |
| `SchemaValidator` | Automatic type-hint and Pydantic-based input/output validation |
| `SemanticValidator` | Register custom semantic validation checks per tool |
| `CustomValidator` | Compose arbitrary validation functions into the pipeline |

### Guardrails — Circuit Breaker, Rate Limiter, Budget Control

| Component | Description |
|---|---|
| `CircuitBreaker` | CLOSED → OPEN → HALF_OPEN state machine to prevent cascading failures |
| `RateLimiter` | Token bucket with per-second/minute/hour rate limiting |
| `TokenBudget` | Per-call and per-session cost and call-count budget enforcement |
| `RetryPolicy` | Exponential backoff with jitter and configurable exception filtering |
| `timeout` | Thread-based (sync) and asyncio (async) timeout enforcement |

### Testing — Trace Recording and Test Generation

| Component | Description |
|---|---|
| `TraceRecorder` | Context manager for recording production agent traces |
| `TraceReplayer` | Replay recorded traces against live tools to detect regressions |
| `TestGenerator` | Auto-generate pytest test files from production traces |
| `assert_tool_call()` | Fluent assertion builder for trace entries |

### Reporting — Metrics and Observability

| Component | Description |
|---|---|
| `ConsoleReporter` | Rich-powered colour terminal tables |
| `JsonReporter` | JSON reports with latency percentiles and anomaly detection |

## Comparison with Other AI Agent Safety Tools

| Feature | agentguard | guardrails-ai | NeMo Guardrails | AgentCircuit | Langfuse | LangSmith |
|---|---|---|---|---|---|---|
| Response anomaly detection | ✅ Multi-signal | ❌ Text-only | ❌ | ❌ | ❌ | ❌ |
| Tool call input/output validation | ✅ Type hints + Pydantic | ✅ Validators | ❌ | ✅ Pydantic | ❌ | ❌ |
| Framework-agnostic | ✅ Any function | ✅ | ✅ | ✅ | ✅ | ❌ LangChain-first |
| Circuit breaker | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Rate limiting | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Budget enforcement | ✅ Per-call + session | ❌ | ❌ | ✅ Global | Token tracking | Token tracking |
| Auto test generation | ✅ From traces | ❌ | ❌ | ❌ | ❌ | ❌ |
| Zero dependencies* | ✅ pydantic only | ❌ Many | ❌ NVIDIA stack | ❌ | ❌ | ❌ |
| Self-hosted | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Open source | ✅ MIT | ✅ | ✅ Apache | ✅ MIT | ✅ | ❌ |

*Core library requires only `pydantic>=2.0`. No NVIDIA dependencies, no cloud services, no API keys needed.

## Who Is This For?

- **AI/ML Engineers** building production agent systems with OpenAI, Anthropic, or open-source LLMs
- **Backend Developers** adding LLM-powered features who need reliability guarantees
- **Platform Teams** managing multi-agent deployments with cost and safety concerns
- **Researchers** studying agent reliability, response integrity, and tool-call verification

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
git clone https://github.com/rigvedrs/agentguard.git
cd agentguard
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Stop trusting your AI agents blindly.</b><br>
  <a href="https://github.com/rigvedrs/agentguard">⭐ Star on GitHub</a> · <a href="https://pypi.org/project/awesome-agentguard/">Install from PyPI</a> · <a href="https://github.com/rigvedrs/agentguard/issues">Report a Bug</a> · <a href="https://github.com/rigvedrs/agentguard/issues">Request a Feature</a>
</p>
