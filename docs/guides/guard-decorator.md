# The @guard Decorator

The `@guard` decorator is agentguard's primary interface. It wraps any Python callable with the full protection stack: validation, hallucination detection, circuit breaking, rate limiting, budget enforcement, retries, timeouts, and trace recording.

## Basic Usage

### Zero-config

```python
from agentguard import guard

@guard
def fetch_data(url: str) -> dict:
    import requests
    return requests.get(url).json()
```

### With keyword arguments

```python
@guard(validate_input=True, max_retries=3, timeout=30.0)
def fetch_data(url: str) -> dict:
    import requests
    return requests.get(url).json()
```

### With a `GuardConfig` object

```python
from agentguard import guard, GuardConfig

config = GuardConfig(
    validate_input=True,
    validate_output=True,
    max_retries=3,
    timeout=30.0,
    record=True,
)

@guard(config=config)
def fetch_data(url: str) -> dict:
    import requests
    return requests.get(url).json()
```

### Applied programmatically (no decorator syntax)

```python
from agentguard.core.guard import GuardedTool

guarded = GuardedTool(fetch_data, config=config)
result = guarded(url="https://api.example.com/data")
```

---

## All Configuration Options

### Validation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `validate_input` | `bool` | `False` | Validate function arguments against type hints |
| `validate_output` | `bool` | `False` | Validate the return value against the declared return type |
| `verify_response` | `bool` | `False` | Run response anomaly detection (timing, schema, patterns, values) |
| `detect_hallucination` | `bool` | `False` | Deprecated alias for `verify_response` |

```python
@guard(
    validate_input=True,    # Check args before calling the function
    validate_output=True,   # Check return value after calling
    verify_response=True,   # Check timing, schema, patterns, statistical anomalies
)
def get_stock_price(ticker: str) -> dict:
    ...
```

### Retries

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_retries` | `int` | `0` | Simple retry count (exponential backoff) |
| `retry` | `RetryConfig \| None` | `None` | Fine-grained retry configuration |

```python
from agentguard import guard
from agentguard.core.types import RetryConfig

# Simple
@guard(max_retries=3)
def call_api(): ...

# Fine-grained
@guard(retry=RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True,
    retryable_exceptions=(ConnectionError, TimeoutError),
))
def call_api(): ...
```

With `RetryConfig.retryable_exceptions`, only those exception types trigger a retry. Empty tuple (default) means retry on any exception.

### Timeout

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timeout` | `float \| None` | `None` | Timeout in seconds. `None` = no timeout |
| `timeout_config` | `TimeoutConfig \| None` | `None` | Fine-grained timeout config |

```python
from agentguard.core.types import TimeoutConfig, GuardAction

@guard(timeout=10.0)  # Raises TimeoutError after 10 seconds

@guard(timeout_config=TimeoutConfig(
    timeout_seconds=10.0,
    on_timeout=GuardAction.WARN,  # Log but don't raise
))
def slow_api(): ...
```

Sync functions use a background thread for timeout enforcement. Async functions use `asyncio.wait_for`.

### Budget Enforcement

| Parameter | Type | Default | Description |
|---|---|---|---|
| `budget` | `BudgetConfig \| None` | `None` | Cost and call-count budgets |

```python
from agentguard import TokenBudget

@guard(budget=TokenBudget(
    max_cost_per_session=5.00,
    max_calls_per_session=100,
    alert_threshold=0.80,   # Warn at 80% usage
).config)
def call_llm_api(prompt: str) -> str: ...
```

Or build the config directly:

```python
from agentguard.core.types import BudgetConfig, GuardAction

@guard(budget=BudgetConfig(
    max_cost_per_call=0.10,
    max_cost_per_session=5.00,
    max_calls_per_session=100,
    alert_threshold=0.80,
    on_exceed=GuardAction.BLOCK,  # BLOCK, WARN, or LOG
    cost_per_call=0.001,          # Fixed cost per call when dynamic pricing unavailable
))
def expensive_tool(): ...
```

### Rate Limiting

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rate_limit` | `RateLimitConfig \| None` | `None` | Token bucket rate limiter |

```python
from agentguard import RateLimiter

@guard(rate_limit=RateLimiter(calls_per_minute=30).config)
def search_api(query: str): ...

# Fine-grained
from agentguard.core.types import RateLimitConfig

@guard(rate_limit=RateLimitConfig(
    calls_per_second=2.0,
    calls_per_minute=60.0,
    calls_per_hour=500.0,
    burst=5,                     # Allow burst of 5 before limiting
    on_limit=GuardAction.BLOCK,
))
def search_api(query: str): ...
```

### Circuit Breaker

| Parameter | Type | Default | Description |
|---|---|---|---|
| `circuit_breaker` | `CircuitBreakerConfig \| None` | `None` | Circuit breaker configuration |

```python
from agentguard import CircuitBreaker

@guard(circuit_breaker=CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
).config)
def external_api(): ...

# Fine-grained
from agentguard.core.types import CircuitBreakerConfig

@guard(circuit_breaker=CircuitBreakerConfig(
    failure_threshold=5,     # Open after 5 consecutive failures
    recovery_timeout=60.0,   # Wait 60s before probing in HALF_OPEN
    success_threshold=2,     # 2 successes in HALF_OPEN to close
    on_open=GuardAction.BLOCK,
))
def external_api(): ...
```

### Tracing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `record` | `bool` | `False` | Record calls to the trace store |
| `trace_dir` | `str` | `"./traces"` | Directory for trace files |
| `trace_backend` | `str` | `"sqlite"` | Trace backend (`sqlite` or `jsonl`) |
| `trace_db_path` | `str \| None` | `None` | Explicit SQLite database path |
| `session_id` | `str \| None` | `None` | Session grouping identifier |

```python
@guard(
    record=True,
    trace_backend="sqlite",
    trace_dir="./production_traces",
    session_id="user_abc_session_xyz",
)
def query_database(sql: str) -> list[dict]: ...
```

### Hooks

| Parameter | Type | Default | Description |
|---|---|---|---|
| `before_call` | `Callable[[ToolCall], None] \| None` | `None` | Called before tool execution |
| `after_call` | `Callable[[ToolCall, ToolResult], None] \| None` | `None` | Called after tool execution |

```python
from agentguard.core.types import ToolCall, ToolResult
import logging

logger = logging.getLogger(__name__)

def log_before(call: ToolCall) -> None:
    logger.info(f"Calling {call.tool_name} with {call.kwargs}")

def log_after(call: ToolCall, result: ToolResult) -> None:
    logger.info(f"{call.tool_name} completed in {result.execution_time_ms:.1f}ms")

@guard(
    before_call=log_before,
    after_call=log_after,
)
def my_tool(query: str) -> str: ...
```

### Custom Validators

```python
from agentguard.core.types import ValidationResult, ValidatorKind

def no_sql_injection(call) -> ValidationResult:
    sql = call.kwargs.get("sql", "")
    dangerous = ["DROP", "DELETE", "TRUNCATE"]
    for keyword in dangerous:
        if keyword.upper() in sql.upper():
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.CUSTOM,
                message=f"Dangerous SQL keyword detected: {keyword}",
            )
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

@guard(custom_validators=[no_sql_injection])
def run_query(sql: str) -> list[dict]: ...
```

---

## GuardedTool API

The `@guard` decorator returns a `GuardedTool` instance that behaves like the original function but adds extra methods:

### Calling the tool

```python
@guard(validate_input=True)
def my_tool(x: str) -> str:
    return x.upper()

# Synchronous call (works exactly like the original function)
result = my_tool("hello")

# Async call
result = await my_tool.acall("hello")
```

### Accessing metadata

```python
# Original function
my_tool.fn  # The unwrapped callable

# Configuration
my_tool.config  # The GuardConfig used

# Call statistics
registry = my_tool.registry_entry  # ToolRegistration with stats
print(registry.call_count)
print(registry.failure_count)
print(registry.avg_latency_ms)
```

---

## Common Patterns

### Shared config across a module

```python
# config.py
from agentguard import GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig

PRODUCTION_GUARD = GuardConfig(
    validate_input=True,
    validate_output=True,
    detect_hallucination=True,
    max_retries=3,
    timeout=30.0,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    rate_limit=RateLimitConfig(calls_per_minute=60),
    record=True,
)

# tools.py
from agentguard import guard
from .config import PRODUCTION_GUARD

@guard(config=PRODUCTION_GUARD)
def search_web(query: str) -> str: ...

@guard(config=PRODUCTION_GUARD)
def query_db(sql: str) -> list[dict]: ...
```

### Different configs per environment

```python
import os
from agentguard import GuardConfig

def make_guard_config() -> GuardConfig:
    if os.getenv("ENV") == "production":
        return GuardConfig(
            validate_input=True,
            detect_hallucination=True,
            max_retries=3,
            record=True,
        )
    else:
        # Faster in development
        return GuardConfig(validate_input=True)

config = make_guard_config()

@guard(config=config)
def my_tool(x: str) -> str: ...
```

### Disabling a guard for testing

```python
import os
from agentguard import guard, GuardConfig

test_config = GuardConfig()  # Zero-config, no retries, no circuit breaker

@guard(config=GuardConfig() if os.getenv("TESTING") else PRODUCTION_GUARD)
def my_tool(x: str) -> str: ...
```

---

## Troubleshooting

### `ValidationError: argument 'limit' expected int, got str`

The AI agent passed a string where an integer was expected. Set `validate_input=True` to catch this before it reaches your function. To see which agent prompt caused the bad call, enable `record=True` and inspect the trace.

### `CircuitOpenError: Circuit breaker for 'my_tool' is OPEN`

The circuit breaker opened because `failure_threshold` consecutive failures occurred. The circuit will probe again after `recovery_timeout` seconds. Check your external dependency's health. You can reset manually:

```python
from agentguard.core.registry import global_registry
tool_reg = global_registry.get("my_tool")
if tool_reg and tool_reg.circuit_breaker:
    tool_reg.circuit_breaker.reset()
```

### Retries aren't triggering

By default, `RetryConfig.retryable_exceptions` is empty, which means retry on any exception. If you set specific exception types and your exception doesn't match, no retry occurs. Check the exception type with:

```python
try:
    result = my_tool()
except Exception as e:
    print(type(e).__qualname__)
```

### `TimeoutError` on every call

Your function takes longer than `timeout` seconds. Either increase the timeout or optimize the function. To diagnose, temporarily remove the timeout and check `ToolResult.execution_time_ms` in your traces.
