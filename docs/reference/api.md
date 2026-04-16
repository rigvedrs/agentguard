# API Reference

Complete reference for all public classes and functions in agentguard.

---

## Core

### `guard`

```python
from agentguard import guard
```

The primary decorator. Wraps any callable with the full agentguard protection stack.

**Signatures:**

```python
# Zero-config (no parentheses)
@guard
def my_tool(x: str) -> str: ...

# With keyword arguments
@guard(validate_input=True, max_retries=3, timeout=30.0, record=True)
def my_tool(x: str) -> str: ...

# With a GuardConfig object
@guard(config=GuardConfig(validate_input=True))
def my_tool(x: str) -> str: ...

# Programmatic application
from agentguard.core.guard import GuardedTool
guarded = GuardedTool(my_fn, config=GuardConfig())
```

**Returns:** `GuardedTool`

All `GuardConfig` fields can be passed as keyword arguments to `@guard`.

---

### `GuardedTool`

```python
from agentguard.core.guard import GuardedTool
```

The wrapper object created by `@guard`. Behaves like the original function.

| Attribute | Type | Description |
|---|---|---|
| `fn` | `Callable` | The original unwrapped function |
| `config` | `GuardConfig` | The applied guard configuration |
| `__name__` | `str` | Name of the original function |
| `__doc__` | `str` | Docstring of the original function |

| Method | Description |
|---|---|
| `__call__(*args, **kwargs)` | Synchronous execution |
| `acall(*args, **kwargs)` | Async execution (returns coroutine) |

---

### `GuardConfig`

```python
from agentguard import GuardConfig
from agentguard.core.types import GuardConfig
```

Configuration dataclass for `@guard`. All fields are optional.

| Field | Type | Default | Description |
|---|---|---|---|
| `validate_input` | `bool` | `False` | Validate arguments against type hints |
| `validate_output` | `bool` | `False` | Validate return value against return type |
| `detect_hallucination` | `bool` | `False` | Run hallucination detection |
| `max_retries` | `int` | `0` | Simple retry count |
| `retry` | `RetryConfig \| None` | `None` | Fine-grained retry config |
| `timeout` | `float \| None` | `None` | Timeout in seconds |
| `timeout_config` | `TimeoutConfig \| None` | `None` | Fine-grained timeout config |
| `budget` | `BudgetConfig \| None` | `None` | Cost/call budget |
| `rate_limit` | `RateLimitConfig \| None` | `None` | Rate limiting |
| `circuit_breaker` | `CircuitBreakerConfig \| None` | `None` | Circuit breaker |
| `record` | `bool` | `False` | Write traces to the configured store |
| `trace_dir` | `str` | `"./traces"` | Trace directory or SQLite parent directory |
| `trace_backend` | `str` | `"sqlite"` | Trace backend: `sqlite` or `jsonl` |
| `trace_db_path` | `str \| None` | `None` | Explicit SQLite database path |
| `session_id` | `str \| None` | `None` | Session identifier |
| `custom_validators` | `list[Callable]` | `[]` | Custom validator functions |
| `before_call` | `Callable \| None` | `None` | Pre-call hook |
| `after_call` | `Callable \| None` | `None` | Post-call hook |

---

### `ToolRegistry`

```python
from agentguard.core.registry import global_registry, ToolRegistry
```

Global registry of all guarded tools. Tools are registered automatically when `@guard` is applied.

| Method | Description |
|---|---|
| `get(name) -> ToolRegistration \| None` | Get registration by tool name |
| `all() -> dict[str, ToolRegistration]` | All registered tools |
| `stats() -> dict` | Aggregated statistics |
| `health() -> dict` | Circuit breaker states and failure counts |

---

## Validators

### `HallucinationDetector`

```python
from agentguard import HallucinationDetector
```

Multi-signal hallucination detector.

```python
detector = HallucinationDetector(threshold=0.6, weights=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | `0.6` | Minimum confidence to flag as hallucinated (0–1) |
| `weights` | `dict \| None` | `None` | Override signal weights |

**Methods:**

```python
detector.register_tool(
    name: str,
    expected_latency_ms: tuple[float, float] | None = None,
    required_fields: list[str] | None = None,
    response_patterns: list[str] | None = None,
    forbidden_patterns: list[str] | None = None,
)

result: HallucinationResult = detector.verify(
    tool_name: str,
    execution_time_ms: float,
    response: Any,
)
```

---

### `SchemaValidator`

```python
from agentguard.validators import SchemaValidator
```

Type-hint and Pydantic-based input/output validation. Used internally by `@guard(validate_input=True)`.

---

### `SemanticValidator`

```python
from agentguard.validators import SemanticValidator
```

Register named validation rules per tool:

```python
validator = SemanticValidator()
validator.register("my_tool", lambda call: ValidationResult(...))
```

---

## Guardrails

### `CircuitBreaker`

```python
from agentguard import CircuitBreaker
```

Convenience wrapper that creates a `CircuitBreakerConfig`:

```python
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
config = cb.config  # CircuitBreakerConfig
```

---

### `RateLimiter`

```python
from agentguard import RateLimiter
```

Convenience wrapper for `RateLimitConfig`:

```python
rl = RateLimiter(calls_per_minute=60, calls_per_second=None)
config = rl.config  # RateLimitConfig
```

---

### `TokenBudget`

```python
from agentguard import TokenBudget
```

Convenience wrapper for `BudgetConfig`:

```python
budget = TokenBudget(
    max_cost_per_session=5.00,
    max_calls_per_session=100,
    alert_threshold=0.80,
)
config = budget.config  # BudgetConfig
```

For response-based LLM cost tracking, pair `TokenBudget` with the tracked client helpers:

```python
from agentguard.integrations import (
    guard_openai_client,
    guard_openai_compatible_client,
    guard_anthropic_client,
)
```

---

## Tracing

### `record_session`

```python
from agentguard import record_session
```

Context manager that records all guarded tool calls within the block:

```python
with record_session("./traces", backend="sqlite", session_id="optional") as recorder:
    result = my_tool("value")

# recorder.entries → list[TraceEntry]
```

---

### `TraceRecorder`

```python
from agentguard import TraceRecorder
```

Manual recorder that can be started and stopped:

```python
recorder = TraceRecorder("./traces")
recorder.start()
result = my_tool("value")
recorder.stop()
entries = recorder.entries
```

---

## LLM Cost Tracking

### `guard_openai_client`

Wrap an OpenAI client and record usage-based spend from chat completions and responses API calls.

### `guard_openai_compatible_client`

Wrap an OpenAI-compatible client and resolve spend through the compatible usage extractor registry.

### `guard_anthropic_client`

Wrap an Anthropic client and record spend from Claude message responses.

### `InMemoryCostLedger` / `NullCostLedger`

Ledger implementations for retaining or discarding `LLMSpendEvent` records.

---

## Testing

### `assert_tool_call`

```python
from agentguard import assert_tool_call
```

Fluent assertion builder for `TraceEntry` objects:

```python
assert_tool_call(entry) \
    .succeeded() \
    .within_ms(5000) \
    .returned_dict() \
    .has_keys("temperature", "humidity")
```

---

### `TestGenerator`

```python
from agentguard.testing import TestGenerator
```

Generate pytest test files from trace files:

```python
gen = TestGenerator(
    traces_dir="./traces",
    include_tools=None,       # list[str] or None for all
    exclude_tools=None,
    max_tests_per_tool=None,  # int or None for unlimited
)
gen.generate_tests("tests/test_generated.py")
```

---

### `TraceReplayer`

```python
from agentguard.testing import TraceReplayer
```

Replay recorded traces against a live implementation:

```python
replayer = TraceReplayer(traces_dir="./traces")

results = replayer.replay_all(
    tools={"get_weather": get_weather},
    comparator=None,  # optional custom comparison function
)

results = replayer.replay_session(
    session_id="session_001",
    tools={"get_weather": get_weather},
)
```

---

## Reporting

### `ConsoleReporter`

```python
from agentguard.reporting import ConsoleReporter
```

Rich-powered terminal tables:

```python
reporter = ConsoleReporter()
reporter.report(entries)  # Prints a formatted table
```

---

### `JsonReporter`

```python
from agentguard.reporting import JsonReporter
```

JSON report with latency percentiles:

```python
reporter = JsonReporter()
report = reporter.report(entries)  # Returns dict
reporter.write(entries, "report.json")
```

---

## Integrations

| Symbol | Module |
|---|---|
| `OpenAIToolExecutor` | `agentguard.integrations.openai_integration` |
| `guard_openai_tools` | `agentguard.integrations.openai_integration` |
| `function_to_openai_tool` | `agentguard.integrations.openai_integration` |
| `execute_openai_tool_call` | `agentguard.integrations.openai_integration` |
| `AnthropicToolExecutor` | `agentguard.integrations.anthropic_integration` |
| `guard_anthropic_tools` | `agentguard.integrations.anthropic_integration` |
| `function_to_anthropic_tool` | `agentguard.integrations.anthropic_integration` |
| `GuardedLangChainTool` | `agentguard.integrations.langchain_integration` |
| `guard_langchain_tools` | `agentguard.integrations.langchain_integration` |
| `GuardedMCPServer` | `agentguard.integrations.mcp_integration` |
| `GuardedMCPClient` | `agentguard.integrations.mcp_integration` |
| `GuardedCrewAITool` | `agentguard.integrations.crewai_integration` |
| `guard_crewai_tools` | `agentguard.integrations.crewai_integration` |
| `GuardedAutoGenTool` | `agentguard.integrations.autogen_integration` |
| `guard_autogen_tool` | `agentguard.integrations.autogen_integration` |
| `guard_autogen_tools` | `agentguard.integrations.autogen_integration` |
| `register_guarded_tools` | `agentguard.integrations.autogen_integration` |
| `Provider` | `agentguard.integrations.openai_compatible` |
| `Providers` | `agentguard.integrations.openai_compatible` |
| `guard_tools` | `agentguard.integrations.openai_compatible` |
| `create_client` | `agentguard.integrations.openai_compatible` |

All of the above are also importable from `agentguard.integrations`.
