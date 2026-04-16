# GuardConfig Reference

`GuardConfig` is the single configuration object for `@guard`. All fields are optional with sensible defaults.

```python
from agentguard import GuardConfig
# or
from agentguard.core.types import GuardConfig
```

---

## Field Reference

### Validation Fields

#### `validate_input: bool = False`

Validate function arguments against Python type hints before the function runs.

```python
@guard(validate_input=True)
def add(a: int, b: int) -> int:
    return a + b

add(a="two", b=3)  # Raises ValidationError
```

Works with: `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, Pydantic models, `TypedDict`, `Optional[X]`, `Union[X, Y]`.

---

#### `validate_output: bool = False`

Validate the function's return value against its declared return type.

```python
@guard(validate_output=True)
def get_user(id: int) -> dict:
    return [1, 2, 3]  # Raises ValidationError — expected dict, got list
```

---

#### `detect_hallucination: bool = False`

Run multi-signal hallucination detection on every call result.

When enabled, the guard checks execution time, schema compliance, and response patterns. Detected hallucinations raise `HallucinationError` (by default).

---

### Resilience Fields

#### `max_retries: int = 0`

Number of automatic retry attempts on failure. `0` means no retries.

Uses exponential backoff: delays of `1s`, `2s`, `4s`, ... up to `max_delay` with jitter.

For fine-grained control, use `retry` (see below).

---

#### `retry: RetryConfig | None = None`

Fine-grained retry configuration. Overrides `max_retries` when set.

```python
from agentguard.core.types import RetryConfig

GuardConfig(retry=RetryConfig(
    max_retries=5,
    initial_delay=0.5,   # seconds
    max_delay=30.0,      # seconds
    backoff_factor=2.0,  # multiplier per attempt
    jitter=True,         # randomise delay
    retryable_exceptions=(ConnectionError, TimeoutError),
))
```

`retryable_exceptions`: only these exception types trigger a retry. If empty, any exception triggers a retry.

---

#### `timeout: float | None = None`

Maximum execution time in seconds. Raises `TimeoutError` when exceeded.

Sync functions: enforced via a background thread (blocking timeout).
Async functions: enforced via `asyncio.wait_for`.

---

#### `timeout_config: TimeoutConfig | None = None`

Fine-grained timeout configuration. Overrides `timeout` when set.

```python
from agentguard.core.types import TimeoutConfig, GuardAction

GuardConfig(timeout_config=TimeoutConfig(
    timeout_seconds=10.0,
    on_timeout=GuardAction.WARN,  # Don't raise — just warn
))
```

---

### Budget Fields

#### `budget: BudgetConfig | None = None`

Cost and call-count budget enforcement.

```python
from agentguard.core.types import BudgetConfig, GuardAction

GuardConfig(budget=BudgetConfig(
    max_cost_per_call=0.10,          # Max $0.10 per single call
    max_cost_per_session=5.00,       # Max $5.00 total for session
    max_calls_per_session=100,       # Max 100 calls per session
    alert_threshold=0.80,            # Warn at 80% usage
    on_exceed=GuardAction.BLOCK,     # Block when exceeded
    cost_per_call=0.001,             # Explicit fallback if dynamic pricing unavailable
    use_dynamic_llm_costs=True,      # Track real LLM cost from provider responses
    model_pricing_overrides={"my-model": (1.0, 3.0)},
    record_llm_spend=True,
))
```

Convenience: `TokenBudget(max_cost_per_session=5.00).config`.

`model_pricing_overrides` values are `(input_dollars_per_1m, output_dollars_per_1m)`. When present, override pricing is used before LiteLLM. `cost_ledger` can be set to an implementation such as `InMemoryCostLedger` to retain spend events for reporting beyond the in-memory budget counters.

---

### Rate Limiting Fields

#### `rate_limit: RateLimitConfig | None = None`

Token bucket rate limiter.

```python
from agentguard.core.types import RateLimitConfig, GuardAction

GuardConfig(rate_limit=RateLimitConfig(
    calls_per_second=2.0,
    calls_per_minute=60.0,
    calls_per_hour=500.0,
    burst=5,
    on_limit=GuardAction.BLOCK,
    shared_key=None,
))
```

Convenience: `RateLimiter(calls_per_minute=60).config`.

`shared_key=None` shares by tool name, `shared_key=""` creates a per-instance
bucket, and any non-empty string creates a custom shared group.

---

### Circuit Breaker Fields

#### `circuit_breaker: CircuitBreakerConfig | None = None`

Circuit breaker configuration.

```python
from agentguard.core.types import CircuitBreakerConfig, GuardAction

GuardConfig(circuit_breaker=CircuitBreakerConfig(
    failure_threshold=5,        # Open after 5 consecutive failures
    recovery_timeout=60.0,      # Wait 60s before probing
    success_threshold=1,        # 1 success in HALF_OPEN to close
    on_open=GuardAction.BLOCK,  # Block when OPEN
))
```

Convenience: `CircuitBreaker(failure_threshold=5, recovery_timeout=60).config`.

---

### Tracing Fields

#### `record: bool = False`

Write every call to the configured trace store.

---

#### `trace_dir: str = "./traces"`

Directory used for legacy JSONL traces or as the parent directory for the default SQLite database.

---

#### `trace_backend: str = "sqlite"`

Trace persistence backend. Use `"sqlite"` for the production default or `"jsonl"` for legacy file-backed traces.

---

#### `trace_db_path: str | None = None`

Explicit SQLite database path. When unset, agentguard stores traces in `trace_dir/agentguard_traces.db`.

---

#### `session_id: str | None = None`

Identifier for grouping related calls in a trace session. If `None`, a default session identifier is used.

---

### Hook Fields

#### `before_call: Callable[[ToolCall], None] | None = None`

Called immediately before the tool function runs. Receives the `ToolCall` record.

```python
def log_call(call: ToolCall) -> None:
    logger.info(f"→ {call.tool_name}({call.kwargs})")

GuardConfig(before_call=log_call)
```

Raising an exception in `before_call` aborts the tool call.

---

#### `after_call: Callable[[ToolCall, ToolResult], None] | None = None`

Called immediately after the tool function completes (success or failure). Receives both the `ToolCall` and the `ToolResult`.

```python
def update_metrics(call: ToolCall, result: ToolResult) -> None:
    metrics.record(call.tool_name, result.execution_time_ms, result.status)

GuardConfig(after_call=update_metrics)
```

---

### Validator Fields

#### `custom_validators: list[Callable] = []`

List of custom validator callables. Each must return a `ValidationResult`.

```python
def no_dangerous_sql(call: ToolCall) -> ValidationResult:
    if "DROP" in call.kwargs.get("sql", "").upper():
        return ValidationResult(valid=False, kind=ValidatorKind.CUSTOM, message="Dangerous SQL")
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

GuardConfig(custom_validators=[no_dangerous_sql])
```

---

## Complete Example

```python
from agentguard import GuardConfig
from agentguard.core.types import (
    RetryConfig,
    TimeoutConfig,
    BudgetConfig,
    RateLimitConfig,
    CircuitBreakerConfig,
    GuardAction,
)

config = GuardConfig(
    # Validation
    validate_input=True,
    validate_output=True,
    detect_hallucination=True,

    # Retries
    retry=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
        retryable_exceptions=(ConnectionError,),
    ),

    # Timeout
    timeout=30.0,

    # Budget
    budget=BudgetConfig(
        max_cost_per_session=5.00,
        max_calls_per_session=100,
        alert_threshold=0.80,
    ),

    # Rate limit
    rate_limit=RateLimitConfig(
        calls_per_minute=60,
        burst=5,
    ),

    # Circuit breaker
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60,
    ),

    # Tracing
    record=True,
    trace_dir="./production_traces",
    session_id="session_001",
)
```
