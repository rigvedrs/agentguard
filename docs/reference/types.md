# Types Reference

Core data models used throughout agentguard.

```python
from agentguard.core.types import (
    ToolCall, ToolResult, TraceEntry, ValidationResult,
    HallucinationResult, GuardAction, CircuitState,
    ToolCallStatus, ValidatorKind,
)
```

---

## `ToolCall`

A record of a tool invocation, captured before execution.

```python
class ToolCall(BaseModel):
    call_id: str              # UUID — unique per invocation
    tool_name: str            # Name of the function/tool
    args: tuple[Any, ...]     # Positional arguments
    kwargs: dict[str, Any]    # Keyword arguments
    timestamp: datetime       # UTC time when the call was initiated
    session_id: str | None    # Optional session grouping ID
    metadata: dict[str, Any]  # Arbitrary user metadata
```

**Usage in hooks and validators:**

```python
def before_call(call: ToolCall) -> None:
    print(f"{call.tool_name}({call.kwargs}) at {call.timestamp}")

def my_validator(call: ToolCall) -> ValidationResult:
    sql = call.kwargs.get("sql", "")
    ...
```

---

## `ToolResult`

A record of a completed tool invocation.

```python
class ToolResult(BaseModel):
    call_id: str                          # Matches ToolCall.call_id
    tool_name: str
    status: ToolCallStatus                # SUCCESS, FAILURE, TIMEOUT, etc.
    return_value: Any                     # Tool return value, or None on failure
    exception: str | None                 # Exception message if failed
    exception_type: str | None            # Fully qualified exception class name
    execution_time_ms: float              # Wall-clock time in milliseconds
    retry_count: int                      # Number of retries performed
    validations: list[ValidationResult]   # All validation checks applied
    hallucination: HallucinationResult | None  # Hallucination check result
    cost: float | None                    # Estimated cost in USD
    timestamp: datetime                   # UTC time when call completed
```

**Properties:**

```python
result.succeeded  # True if status == SUCCESS
result.failed     # True if status is any failure state
```

---

## `TraceEntry`

Combined record of a call and its result, stored in trace files.

```python
class TraceEntry(BaseModel):
    call: ToolCall
    result: ToolResult
```

**Properties:**

```python
entry.call_id    # Shortcut to call.call_id
entry.tool_name  # Shortcut to call.tool_name
```

---

## `ValidationResult`

Result of a single validator check.

```python
class ValidationResult(BaseModel):
    valid: bool
    kind: ValidatorKind       # SCHEMA, HALLUCINATION, SEMANTIC, or CUSTOM
    message: str              # Human-readable description
    details: dict[str, Any]   # Structured debugging information
```

---

## `HallucinationResult`

Result of hallucination detection analysis.

```python
class HallucinationResult(BaseModel):
    is_hallucinated: bool
    confidence: float         # 0.0 = definitely real, 1.0 = definitely hallucinated
    reason: str               # Human-readable explanation
    signals: dict[str, Any]   # Raw signal values used in scoring
```

---

## Enums

### `ToolCallStatus`

```python
class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "rate_limited"
    VALIDATION_FAILED = "validation_failed"
    HALLUCINATED = "hallucinated"
    RETRIED = "retried"
```

---

### `GuardAction`

```python
class GuardAction(str, Enum):
    BLOCK = "block"   # Raise an exception and abort
    WARN = "warn"     # Log a warning but allow the call
    LOG = "log"       # Silently record, allow the call
```

Used in `CircuitBreakerConfig.on_open`, `BudgetConfig.on_exceed`, `RateLimitConfig.on_limit`, `TimeoutConfig.on_timeout`.

---

### `CircuitState`

```python
class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Blocking all calls
    HALF_OPEN = "half_open"  # Testing recovery
```

---

### `ValidatorKind`

```python
class ValidatorKind(str, Enum):
    SCHEMA = "schema"
    HALLUCINATION = "hallucination"
    SEMANTIC = "semantic"
    CUSTOM = "custom"
```

---

## Config Types

### `RetryConfig`

```python
class RetryConfig(BaseModel):
    max_retries: int = 3
    initial_delay: float = 1.0      # seconds
    max_delay: float = 60.0         # seconds
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = ()
```

### `TimeoutConfig`

```python
class TimeoutConfig(BaseModel):
    timeout_seconds: float
    on_timeout: GuardAction = GuardAction.BLOCK
```

### `BudgetConfig`

```python
class BudgetConfig(BaseModel):
    max_cost_per_call: float | None = None
    max_cost_per_session: float | None = None
    max_calls_per_session: int | None = None
    alert_threshold: float = 0.80
    on_exceed: GuardAction = GuardAction.BLOCK
    cost_per_call: float | None = None
    use_dynamic_llm_costs: bool = True
    model_pricing_overrides: dict[str, tuple[float, float]] | None = None
    record_llm_spend: bool = True
    cost_ledger: CostLedger | None = None
```

### `UsageKind`

```python
class UsageKind(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    MULTIMODAL = "multimodal"
```

### `LLMUsage`

Normalized provider-reported usage payload for model calls.

### `LLMCostBreakdown`

Resolved pricing metadata for an LLM call, including `pricing_source`, `priced_model`, and `pricing_as_of`.

### `LLMSpendEvent`

The persisted/logged spend record written to traces, telemetry, and optional ledgers.

### `RateLimitConfig`

```python
class RateLimitConfig(BaseModel):
    calls_per_second: float | None = None
    calls_per_minute: float | None = None
    calls_per_hour: float | None = None
    burst: int = 1
    on_limit: GuardAction = GuardAction.BLOCK
    shared_key: str | None = None
```

`shared_key=None` shares by tool name, `shared_key=""` disables sharing, and
any other string shares a bucket across all tools using that key.

### `CircuitBreakerConfig`

```python
class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 1
    on_open: GuardAction = GuardAction.BLOCK
```
