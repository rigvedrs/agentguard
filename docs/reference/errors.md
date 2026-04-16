# Error Reference

All agentguard exceptions inherit from `AgentGuardError`.

```python
from agentguard.core.types import (
    AgentGuardError,
    ValidationError,
    HallucinationError,
    CircuitOpenError,
    BudgetExceededError,
    RateLimitError,
    TimeoutError,
)
```

---

## Exception Hierarchy

```
AgentGuardError
├── ValidationError
├── HallucinationError
├── CircuitOpenError
├── BudgetExceededError
├── RateLimitError
└── TimeoutError
```

---

## `AgentGuardError`

Base class for all agentguard exceptions. Catch this to handle any agentguard error:

```python
from agentguard.core.types import AgentGuardError

try:
    result = my_tool("value")
except AgentGuardError as e:
    logger.error(f"agentguard blocked the call: {e}")
```

---

## `ValidationError`

Raised when input or output validation fails.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the validation failure |
| `details` | `dict` | Structured details (field name, expected type, received value) |

**When raised:** `validate_input=True` or `validate_output=True` and the validation fails, or a `custom_validator` returns `ValidationResult(valid=False)`.

```python
from agentguard.core.types import ValidationError

try:
    result = add(a="two", b=3)
except ValidationError as e:
    print(e.message)   # "argument 'a' expected int, got str"
    print(e.details)   # {"field": "a", "expected": "int", "got": "str"}
```

---

## `AnomalousResponseError`

Raised when a tool response is flagged as anomalous by the response verifier —
the response violated the expected contract (missing fields, anomalous timing,
pattern mismatch, etc.).

`HallucinationError` is a legacy alias that resolves to the same class.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool flagged |
| `result` | `HallucinationResult` | Full detection result with confidence and signals |

```python
from agentguard import AnomalousResponseError

try:
    result = get_weather("Paris")
except AnomalousResponseError as e:
    print(f"Anomalous response (confidence {e.result.confidence:.2f})")
    print(f"Reason: {e.result.reason}")
    print(f"Signals: {e.result.signals}")
```

---

## `CircuitOpenError`

Raised when the circuit breaker is OPEN and a call arrives.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool with the open circuit |
| `recovery_in` | `float` | Seconds until the next probe attempt |

```python
from agentguard.core.types import CircuitOpenError

try:
    result = call_payment_api(99.99, "tok_visa")
except CircuitOpenError as e:
    print(f"Service down. Retry in {e.recovery_in:.0f}s")
    return {"status": "deferred", "retry_after": e.recovery_in}
```

---

## `BudgetExceededError`

Raised when a call would exceed the configured budget.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool |
| `spent` | `float` | Current spend in USD |
| `limit` | `float` | Budget limit in USD |

```python
from agentguard.core.types import BudgetExceededError

try:
    result = call_llm_api(prompt)
except BudgetExceededError as e:
    print(f"Over budget: ${e.spent:.4f} / ${e.limit:.4f}")
    return {"error": "Budget limit reached for this session"}
```

---

## `RateLimitError`

Raised when a call exceeds the configured rate limit.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool |
| `retry_after` | `float` | Seconds until one token is available |

```python
from agentguard.core.types import RateLimitError
import time

try:
    result = search_api(query)
except RateLimitError as e:
    time.sleep(e.retry_after)
    result = search_api(query)  # Retry after waiting
```

---

## `TimeoutError`

Raised when a tool call exceeds its configured timeout.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tool_name` | `str` | Name of the tool that timed out |
| `timeout` | `float` | The timeout limit that was exceeded |

```python
from agentguard.core.types import TimeoutError

try:
    result = slow_api()
except TimeoutError as e:
    print(f"{e.tool_name} timed out after {e.timeout:.1f}s")
    return {"error": "The request took too long. Please try again."}
```

---

## Catching All Guard Errors

```python
from agentguard.core.types import (
    AgentGuardError,
    CircuitOpenError,
    BudgetExceededError,
    RateLimitError,
)

def safe_call(tool, *args, **kwargs):
    try:
        return tool(*args, **kwargs)
    except CircuitOpenError as e:
        return {"error": f"service_unavailable", "retry_in": e.recovery_in}
    except BudgetExceededError:
        return {"error": "budget_exceeded"}
    except RateLimitError as e:
        return {"error": "rate_limited", "retry_in": e.retry_after}
    except AgentGuardError as e:
        return {"error": "guard_error", "message": str(e)}
```
