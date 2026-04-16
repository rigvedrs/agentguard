# Circuit Breaker

## The Problem

Your AI agent calls an external API that starts returning errors. Without protection, the agent will keep retrying — hammering an already-struggling service, spending your budget, and holding up the entire pipeline. When one tool fails, it can cascade into total agent failure.

The **circuit breaker** pattern solves this. Instead of retrying indefinitely, the circuit breaker:

1. Counts consecutive failures
2. Once failures exceed a threshold, **opens the circuit** — immediately rejecting new calls
3. After a recovery timeout, allows a **probe call** to test recovery
4. If the probe succeeds, closes the circuit and resumes normal operation

---

## State Machine

```
              failures >= threshold
CLOSED ─────────────────────────────► OPEN
  │                                     │
  │      success (probe passed)         │ recovery_timeout elapsed
  └─────────────────────────────── HALF_OPEN ◄──────────────────────────┘
```

| State | Behaviour |
|---|---|
| **CLOSED** | Normal operation — all calls pass through |
| **OPEN** | Circuit is open — calls immediately raise `CircuitOpenError` |
| **HALF_OPEN** | One probe call allowed. Success → CLOSED. Failure → OPEN again |

---

## Basic Usage

```python
from agentguard import guard, CircuitBreaker

@guard(circuit_breaker=CircuitBreaker(
    failure_threshold=5,    # Open after 5 consecutive failures
    recovery_timeout=60,    # Wait 60 seconds before probing
).config)
def call_payment_api(amount: float, card_token: str) -> dict:
    """Charge a payment card."""
    import requests
    return requests.post("https://payments.example.com/charge", json={
        "amount": amount,
        "token": card_token,
    }).json()
```

After 5 consecutive failures, `call_payment_api` will raise `CircuitOpenError` immediately (no HTTP request made) until 60 seconds have elapsed and a probe succeeds.

---

## Configuration

```python
from agentguard.core.types import CircuitBreakerConfig, GuardAction

config = CircuitBreakerConfig(
    failure_threshold=5,          # Consecutive failures to open the circuit
    recovery_timeout=60.0,        # Seconds to stay OPEN before probing
    success_threshold=2,          # Successes in HALF_OPEN to close
    on_open=GuardAction.BLOCK,    # BLOCK (default), WARN, or LOG
)
```

### `failure_threshold`

How many consecutive failures must occur before the circuit opens. Choose based on your tolerance for downstream impact:

- **Mission-critical tools** (payments, auth): 2–3
- **General tools** (search, data enrichment): 5–10
- **Idempotent read-only tools** (cache reads): 10–20

### `recovery_timeout`

How long the circuit stays OPEN before allowing a probe call. Choose based on how long the downstream service typically takes to recover:

- **Fast services** (microservices, databases): 30–60s
- **Slow services** (external APIs with incident response): 120–300s
- **Unreliable services**: 300–600s

### `success_threshold`

How many consecutive successes in HALF_OPEN are required to close the circuit. Setting this > 1 provides extra confidence that recovery is stable:

```python
CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=3,  # Require 3 successful probes before declaring recovery
)
```

### `on_open`

What happens when a call arrives at an OPEN circuit:

```python
from agentguard.core.types import GuardAction

# Default: raise CircuitOpenError immediately
on_open=GuardAction.BLOCK

# Log a warning and return None (agent continues with degraded behaviour)
on_open=GuardAction.WARN

# Silently record the block, return None
on_open=GuardAction.LOG
```

---

## Handling `CircuitOpenError`

```python
from agentguard.core.types import CircuitOpenError

try:
    result = call_payment_api(99.99, "tok_visa")
except CircuitOpenError as e:
    print(f"Circuit is open — {e.recovery_in:.0f}s until next probe")
    # Fall back to a queued payment approach, or return an error to the user
    return {"status": "deferred", "retry_after": e.recovery_in}
```

---

## Checking Circuit State

```python
from agentguard.core.registry import global_registry

reg = global_registry.get("call_payment_api")
if reg and reg.circuit_breaker:
    cb = reg.circuit_breaker
    print(f"State: {cb.state}")
    print(f"Failures: {cb.failure_count}")
    print(f"Successes: {cb.success_count}")
```

---

## When to Use Each Pattern

### Use circuit breakers for:

- External HTTP APIs (payment processors, data providers, authentication services)
- Database connections (especially external managed databases)
- Message queue connections
- Any tool that depends on an external service with its own uptime

### Don't use circuit breakers for:

- In-memory operations (list comprehensions, local computation)
- Tools that fail by design for certain inputs (validation functions)
- Tools where you always want to see the error (logging, alerting)

---

## Common Patterns

### Graceful degradation with fallback

```python
from agentguard import guard, CircuitBreaker
from agentguard.core.types import CircuitOpenError

@guard(circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout=30).config)
def live_stock_price(ticker: str) -> dict:
    import requests
    return requests.get(f"https://api.market.com/price/{ticker}").json()

def get_stock_price(ticker: str) -> dict:
    """Get stock price with fallback to cached data."""
    try:
        return live_stock_price(ticker)
    except CircuitOpenError:
        # Serve cached data while the circuit is open
        return cache.get(f"price:{ticker}") or {"ticker": ticker, "price": "unavailable"}
```

### Different thresholds per criticality

```python
from agentguard.core.types import CircuitBreakerConfig

# Critical financial tool — open fast, recover carefully
payment_cb = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=120,
    success_threshold=3,
)

# Non-critical search tool — more tolerant
search_cb = CircuitBreakerConfig(
    failure_threshold=10,
    recovery_timeout=30,
    success_threshold=1,
)
```

### Monitoring circuit state across all tools

```python
from agentguard.core.registry import global_registry

def health_check() -> dict:
    """Return circuit breaker status for all registered tools."""
    status = {}
    for name, reg in global_registry.all().items():
        if reg.circuit_breaker:
            status[name] = {
                "state": reg.circuit_breaker.state.value,
                "failures": reg.circuit_breaker.failure_count,
            }
    return status
```

---

## Troubleshooting

### Circuit opens too quickly

Increase `failure_threshold` or add `max_retries` so transient errors are handled before counting toward the circuit:

```python
@guard(
    max_retries=2,                                          # Retry before counting as failure
    circuit_breaker=CircuitBreaker(failure_threshold=5).config,
)
def my_tool(): ...
```

### Circuit never recovers

Check that your service is actually recovering. A HALF_OPEN probe that fails resets the circuit back to OPEN with a fresh `recovery_timeout`. If your service is still down, the circuit stays open.

### `CircuitOpenError` not caught by outer retry logic

`CircuitOpenError` is deliberately not retryable — there's no point retrying immediately when the circuit is open. Handle it explicitly and return a fallback response or re-raise with a user-friendly message.
