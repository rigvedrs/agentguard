# Rate Limiting

## The Problem

AI agents are prolific tool callers. A single `while True` loop in an agent that calls a search API can exceed your API provider's rate limits in seconds, resulting in HTTP 429 errors, blocked accounts, or surprise bills.

agentguard's rate limiter uses the **token bucket algorithm** to enforce configurable per-second, per-minute, and per-hour limits.

---

## Token Bucket Algorithm

The token bucket works like this:

1. The bucket starts full with `burst` tokens
2. Tokens are replenished continuously at the configured rate
3. Each function call consumes 1 token
4. If the bucket is empty, the call is blocked (or warned, depending on config)

This gives you both a steady-state rate limit *and* a burst allowance for legitimate spikes.

```
Bucket capacity: 5 tokens (burst)
Refill rate: 2 tokens/second (calls_per_second=2.0)

t=0:  [●●●●●] 5 tokens — 5 burst calls allowed immediately
t=1:  [●●●●●] refills to 5 (was 5, still 5)
t=1.5: call → [●●●●] 4 tokens remaining
t=2:  [●●●●●] refilled to 5
```

---

## Basic Usage

```python
from agentguard import guard, RateLimiter

@guard(rate_limit=RateLimiter(calls_per_minute=30).config)
def search_api(query: str) -> list[dict]:
    """Search — limited to 30 calls/minute."""
    import requests
    return requests.get(f"https://search.api.com?q={query}").json()
```

Or use `RateLimitConfig` directly for full control:

```python
from agentguard.core.types import RateLimitConfig, GuardAction

@guard(rate_limit=RateLimitConfig(
    calls_per_second=2.0,
    calls_per_minute=60.0,
    calls_per_hour=500.0,
    burst=5,
    on_limit=GuardAction.BLOCK,
    shared_key=None,
))
def search_api(query: str) -> list[dict]: ...
```

---

## Configuration Reference

### `calls_per_second`

Maximum sustained rate in calls per second. The bucket refills at this rate.

```python
RateLimitConfig(calls_per_second=5.0)   # Up to 5 calls/second sustained
```

### `calls_per_minute`

Convenience shorthand — converted to `calls_per_second = calls_per_minute / 60`.

```python
RateLimitConfig(calls_per_minute=120)   # 2 calls/second
```

### `calls_per_hour`

For very slow-refilling buckets.

```python
RateLimitConfig(calls_per_hour=1000)    # ~0.278 calls/second
```

You can combine limits. The most restrictive is applied:

```python
RateLimitConfig(
    calls_per_second=5.0,    # No more than 5/second
    calls_per_hour=1000.0,   # No more than 1000/hour
)
```

### `burst`

Maximum tokens in the bucket (also the starting token count). Allows short bursts above the sustained rate.

```python
# Allow up to 10 burst calls, then cap at 2/second
RateLimitConfig(calls_per_second=2.0, burst=10)
```

### `on_limit`

What to do when the bucket is empty:

```python
from agentguard.core.types import GuardAction

on_limit=GuardAction.BLOCK    # Raise RateLimitError (default)
on_limit=GuardAction.WARN     # Log warning, return None
on_limit=GuardAction.LOG      # Silently record, return None
```

### `shared_key`

Controls whether buckets are shared across guarded tool instances:

```python
RateLimitConfig(shared_key=None)          # Default: share by tool name
RateLimitConfig(shared_key="")            # Per-instance bucket
RateLimitConfig(shared_key="provider-x")  # Explicit shared group
```

If multiple tools register the same effective shared key with different rate
limit settings, the first config wins and agentguard emits a warning.

---

## Handling `RateLimitError`

When `on_limit=GuardAction.BLOCK` (the default), exceeding the rate limit raises `RateLimitError`:

```python
from agentguard.core.types import RateLimitError
import time

def search_with_backoff(query: str) -> list[dict]:
    while True:
        try:
            return search_api(query)
        except RateLimitError as e:
            print(f"Rate limited — retrying in {e.retry_after:.1f}s")
            time.sleep(e.retry_after)
```

`RateLimitError.retry_after` is the number of seconds until enough tokens are available for one call.

---

## Common Patterns

### Match your API provider's limits

```python
# OpenAI GPT-4: 500 RPM on tier 1
@guard(rate_limit=RateLimitConfig(calls_per_minute=490, burst=10))
def call_gpt4(prompt: str) -> str: ...

# Anthropic Claude: 1000 RPM
@guard(rate_limit=RateLimitConfig(calls_per_minute=990, burst=20))
def call_claude(prompt: str) -> str: ...

# SerpAPI: 100 searches/month (very slow)
@guard(rate_limit=RateLimitConfig(
    calls_per_hour=3,   # 100/month ≈ 3/hour
    burst=1,
))
def serpapi_search(query: str) -> dict: ...
```

### Per-user rate limiting

By default, rate limits are shared by tool name across `GuardedTool`
instances. If you want isolated buckets per user, set `shared_key=""` when
creating each instance:

```python
from agentguard import guard, GuardConfig
from agentguard.core.types import RateLimitConfig

def create_user_tools(user_id: str) -> dict:
    config = GuardConfig(
        rate_limit=RateLimitConfig(calls_per_minute=10, shared_key=""),
        session_id=user_id,
    )
    from agentguard.core.guard import GuardedTool
    return {
        "search": GuardedTool(search_fn, config=config),
        "query_db": GuardedTool(query_db_fn, config=config),
    }
```

### Shared quota across different tools

Use a custom `shared_key` when different tool names consume the same upstream
provider quota:

```python
provider_limit = RateLimitConfig(calls_per_minute=100, shared_key="serpapi")

@guard(rate_limit=provider_limit)
def web_search(query: str) -> dict: ...

@guard(rate_limit=provider_limit)
def news_search(query: str) -> dict: ...
```

### Graceful degradation instead of error

```python
from agentguard.core.types import GuardAction

@guard(rate_limit=RateLimitConfig(
    calls_per_minute=60,
    on_limit=GuardAction.WARN,   # Return None instead of raising
))
def enrich_lead(email: str) -> dict | None:
    """Enrich a lead — may return None if rate limited."""
    ...

result = enrich_lead("user@example.com")
if result is None:
    # Rate limited — skip enrichment for this lead
    result = {"email": email, "enriched": False}
```

---

## Rate Limiting vs Circuit Breaking

These are complementary, not alternatives:

| | Rate Limiter | Circuit Breaker |
|---|---|---|
| **Purpose** | Prevent *your agent* from calling too fast | Prevent calls to a *failing downstream service* |
| **Triggers on** | Call volume | Failure count |
| **Recovery** | Automatic (bucket refills) | Timed probe |
| **Use when** | You have an API quota | The service is unreliable |

Use both together:

```python
@guard(
    rate_limit=RateLimitConfig(calls_per_minute=60),
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
)
def call_external_api(): ...
```

---

## Troubleshooting

### `RateLimitError` in tests

Tests that call guarded tools rapidly will hit rate limits. Use a permissive config in tests:

```python
# conftest.py
import pytest
from agentguard import GuardConfig

@pytest.fixture
def no_limits():
    return GuardConfig()  # No rate limit

def test_my_tool(no_limits):
    from agentguard.core.guard import GuardedTool
    guarded = GuardedTool(my_fn, config=no_limits)
    for _ in range(100):  # Won't be rate limited
        guarded(arg="value")
```

### Rate limit not enforced

Check that you're using the same `GuardedTool` instance across calls. Each `@guard` application creates a fresh token bucket. If you wrap the function twice, each wrapper has its own bucket.

### Burst calls all succeed, then rate limit kicks in

This is expected behaviour. The burst allows an initial flurry of calls, then enforces the sustained rate. If you want stricter control with no burst, set `burst=1`.
