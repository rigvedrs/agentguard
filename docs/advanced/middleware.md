# Middleware Pipeline

## Overview

agentguard's `middleware.py` provides a composable guard pipeline — a sequence of middleware functions that can be combined, reordered, and extended. This is the underlying mechanism that powers `@guard`.

```python
from agentguard.middleware import GuardMiddleware, MiddlewarePipeline
```

---

## What Is Middleware?

Each guard (validator, rate limiter, circuit breaker, etc.) is a middleware function with the signature:

```python
async def middleware(call: ToolCall, next: Callable) -> ToolResult:
    # Before the call
    ...
    result = await next(call)
    # After the call
    ...
    return result
```

Middleware wraps the next function in the chain. This is the classic "onion" pattern:

```
Request → [Validator] → [RateLimiter] → [CircuitBreaker] → [YourFunction] → ...
                ↑              ↑                ↑                  ↑
            pre-hooks      pre-hooks        pre-hooks           execution
                ↓              ↓                ↓                  ↓
Response ← [Validator] ← [RateLimiter] ← [CircuitBreaker] ← [YourFunction]
```

---

## Built-in Middleware

agentguard ships these middleware components, applied in order:

1. `CircuitBreakerMiddleware` — reject calls when circuit is OPEN
2. `RateLimiterMiddleware` — enforce token bucket rate limits
3. `BudgetMiddleware` — check and update spend budgets
4. `ValidationMiddleware` — validate inputs/outputs
5. `HallucinationMiddleware` — detect hallucinated responses
6. `RetryMiddleware` — retry on failure with backoff
7. `TimeoutMiddleware` — enforce per-call timeout
8. `TraceMiddleware` — record calls to trace files
9. `HookMiddleware` — call `before_call` and `after_call` hooks

---

## Using `MiddlewarePipeline` Directly

For advanced use cases, build a pipeline manually:

```python
from agentguard.middleware import MiddlewarePipeline
from agentguard.core.types import GuardConfig

config = GuardConfig(
    validate_input=True,
    max_retries=2,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
)

pipeline = MiddlewarePipeline(config=config)
result = await pipeline.execute(my_function, args=(), kwargs={"query": "test"})
```

---

## Writing Custom Middleware

```python
from agentguard.middleware import GuardMiddleware
from agentguard.core.types import ToolCall, ToolResult
from typing import Callable, Awaitable

class AuditMiddleware(GuardMiddleware):
    """Log every tool call to an audit trail."""
    
    def __init__(self, audit_service):
        self.audit = audit_service
    
    async def __call__(
        self,
        call: ToolCall,
        next: Callable[[ToolCall], Awaitable[ToolResult]],
    ) -> ToolResult:
        # Record the attempt
        audit_id = self.audit.start(call.tool_name, call.kwargs, call.session_id)
        
        try:
            result = await next(call)
            self.audit.success(audit_id, result.execution_time_ms)
            return result
        except Exception as exc:
            self.audit.failure(audit_id, str(exc))
            raise
```

---

## Composing Pipelines

```python
from agentguard.middleware import MiddlewarePipeline
from agentguard.core.types import GuardConfig

config = GuardConfig(validate_input=True, max_retries=2)

# Add custom middleware
audit = AuditMiddleware(my_audit_service)
pipeline = MiddlewarePipeline(config=config, extra_middleware=[audit])
```

---

## Per-Environment Pipelines

```python
import os

def build_pipeline(config: GuardConfig) -> MiddlewarePipeline:
    extra = []
    
    if os.getenv("ENV") == "production":
        extra.append(AuditMiddleware(audit_service))
        extra.append(TelemetryMiddleware(otel_tracer))
    
    return MiddlewarePipeline(config=config, extra_middleware=extra)
```

---

## Async vs Sync

The pipeline is async-first. For sync functions, `GuardedTool.__call__` runs the async pipeline in a thread:

```python
@guard(validate_input=True)
def sync_tool(x: str) -> str:  # Sync function
    return x.upper()

# Sync call — pipeline runs in a background thread
result = sync_tool("hello")

# Async call — pipeline runs in the event loop
result = await sync_tool.acall("hello")
```
