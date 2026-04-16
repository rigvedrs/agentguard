# Multi-Agent Safety

## Overview

When multiple AI agents collaborate — supervisor/worker hierarchies, parallel research teams, debate patterns — you need safety at the system level, not just the individual tool level. This guide covers patterns for multi-agent deployments with agentguard.

---

## Shared Circuit Breakers

By default, each `GuardedTool` has its own independent circuit breaker. In multi-agent systems, you often want a **shared circuit breaker** so that if Agent A observes 5 consecutive failures, Agent B also stops calling the failing service.

```python
from agentguard.core.guardrails.circuit_breaker import CircuitBreakerState
from agentguard.core.types import GuardConfig, CircuitBreakerConfig
from agentguard.core.guard import GuardedTool

# Create a shared circuit breaker state object
shared_cb = CircuitBreakerState(
    CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
)

# Apply the same circuit breaker to multiple tools
config_with_shared_cb = GuardConfig(circuit_breaker=shared_cb.config)

search_tool = GuardedTool(search_fn, config=config_with_shared_cb)
enrich_tool = GuardedTool(enrich_fn, config=config_with_shared_cb)
# Both tools now share the same circuit
```

---

## Shared Budget Across Agents

Track spending across all agents in a crew with a shared session ID:

```python
from agentguard import GuardConfig
from agentguard.core.types import BudgetConfig
import uuid

# One session ID for the entire crew run
crew_session_id = f"crew_{uuid.uuid4().hex[:8]}"

# Each agent gets the same session_id — budget tracked collectively
def make_agent_config(tool_name: str) -> GuardConfig:
    return GuardConfig(
        validate_input=True,
        budget=BudgetConfig(
            max_cost_per_session=10.00,  # $10 total for all agents
            max_calls_per_session=500,
        ),
        session_id=crew_session_id,     # Shared session
        record=True,
        trace_backend="sqlite",
        trace_dir=f"./traces/{crew_session_id}",
    )
```

---

## Supervisor-Worker Pattern

A supervisor agent coordinates multiple worker agents. Apply guards at both layers:

```python
from agentguard import guard, GuardConfig
from agentguard.core.types import CircuitBreakerConfig, RateLimitConfig

# Supervisor tools — less restrictive (orchestration, not execution)
supervisor_config = GuardConfig(
    validate_input=True,
    max_retries=1,
    timeout=60.0,
)

# Worker tools — stricter (actual external calls)
worker_config = GuardConfig(
    validate_input=True,
    detect_hallucination=True,
    max_retries=3,
    circuit_breaker=CircuitBreakerConfig(failure_threshold=5),
    rate_limit=RateLimitConfig(calls_per_minute=30),
    record=True,
)

# Supervisor tools
@guard(config=supervisor_config)
def delegate_research(topic: str, worker_id: str) -> dict:
    """Delegate a research task to a worker agent."""
    return worker_agents[worker_id].run(topic)

# Worker tools
@guard(config=worker_config)
def search_academic(query: str) -> list[dict]:
    """Search academic databases."""
    return arxiv.search(query)

@guard(config=worker_config)
def summarize_paper(arxiv_id: str) -> str:
    """Summarize an academic paper."""
    return llm.summarize(arxiv.get(arxiv_id))
```

---

## Observing Multi-Agent Traces

With `record=True` and a shared `session_id`, all agent calls end up in the same trace session:

```python
from agentguard.core.trace import TraceStore

store = TraceStore("./traces/crew_abc123", backend="sqlite")
entries = store.read_session("crew_abc123")

# See which agent called which tool
for entry in entries:
    print(f"[{entry.call.metadata.get('agent', '?')}] "
          f"{entry.tool_name}: {entry.result.status}")
```

Attach agent identity to calls via the `before_call` hook:

```python
def tag_with_agent(agent_name: str):
    def hook(call: ToolCall) -> None:
        call.metadata["agent"] = agent_name
    return hook

config = GuardConfig(
    record=True,
    before_call=tag_with_agent("researcher_agent"),
)
```

---

## Rate Limits Across Agents

By default, `@guard` shares rate-limit state across all `GuardedTool`
instances with the same tool name. If multiple agents call the same logical
tool, they automatically draw from one bucket.

```python
from agentguard import guard
from agentguard.core.types import RateLimitConfig

@guard(rate_limit=RateLimitConfig(calls_per_minute=100))
def search_fn(query: str) -> dict:
    ...

# Both agents use guarded instances with the same tool name
agent_a = MyAgent(tools=[search_fn, ...])
agent_b = MyAgent(tools=[search_fn, ...])
```

Set `shared_key=""` if you want per-agent buckets even when names match.
Use a non-empty `shared_key` to share one quota group across different tool
names. If conflicting configs register the same effective key, the first one
wins and agentguard emits a warning.

---

## Preventing Cascading Failures

Multi-agent systems are especially vulnerable to cascading failures. Use circuit breakers at both the tool level and the agent coordination level:

```python
from agentguard import guard
from agentguard.core.types import CircuitBreakerConfig, CircuitOpenError

# Tool level
@guard(circuit_breaker=CircuitBreakerConfig(failure_threshold=3))
def call_worker_agent(task: str, worker_id: str) -> dict:
    return workers[worker_id].execute(task)

# Coordination level — catch circuit open and degrade gracefully
def run_parallel_research(topics: list[str]) -> dict:
    results = {}
    for i, topic in enumerate(topics):
        try:
            results[topic] = call_worker_agent(topic, f"worker_{i % 3}")
        except CircuitOpenError:
            results[topic] = {"status": "deferred", "reason": "worker unavailable"}
    return results
```

---

## Health Dashboard

Aggregate circuit breaker and error statistics across all tools:

```python
from agentguard.core.registry import global_registry

def get_agent_health() -> dict:
    health = {}
    for name, reg in global_registry.all().items():
        health[name] = {
            "calls": reg.call_count,
            "failures": reg.failure_count,
            "error_rate": reg.failure_count / max(reg.call_count, 1),
            "circuit": reg.circuit_breaker.state.value if reg.circuit_breaker else "none",
            "avg_latency_ms": reg.avg_latency_ms,
        }
    return health
```
