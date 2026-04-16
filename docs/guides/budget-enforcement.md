# Budget Enforcement

## The Problem

AI agent costs are notoriously hard to predict. A single prompt change, retry loop, or model escalation can cause an agent to call a paid API 10× more than intended. Without spending limits, you can wake up to an unexpected bill.

agentguard's `TokenBudget` / `BudgetConfig` enforces hard spending limits at the tool level. It is designed to work alongside tool-call validation and response verification, so the same runtime that catches broken tool execution also stops runaway spend.

For LLM API calls, agentguard now also supports real spend tracking by wrapping supported provider clients, reading provider-reported usage from responses, and resolving pricing through LiteLLM when available.

---

## How It Works

The budget enforcer tracks:

- **Per-call cost** — how much this single call costs
- **Session cost** — cumulative cost since the session started
- **Session call count** — number of calls in this session

When any limit is approached (configurable alert threshold) or exceeded, the enforcer takes the configured action (block, warn, or log).

---

## Basic Usage

### Using `TokenBudget` (convenience wrapper)

```python
from agentguard import guard, TokenBudget

@guard(budget=TokenBudget(
    max_cost_per_session=5.00,      # Stop after $5 total
    max_calls_per_session=100,      # Stop after 100 calls
    alert_threshold=0.80,           # Warn at 80% usage
).config)
def call_openai(prompt: str) -> str:
    """Call OpenAI — costs money."""
    import openai
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

### Using `BudgetConfig` directly

```python
from agentguard import guard
from agentguard.core.types import BudgetConfig, GuardAction

@guard(budget=BudgetConfig(
    max_cost_per_call=0.50,         # Each call costs at most $0.50
    max_cost_per_session=10.00,     # Total session budget: $10
    max_calls_per_session=200,      # Max 200 calls per session
    alert_threshold=0.75,           # Alert at 75% of any limit
    on_exceed=GuardAction.BLOCK,    # Block when exceeded
    cost_per_call=0.01,             # Explicit fallback if dynamic pricing unavailable
    use_dynamic_llm_costs=True,     # Enable response-based LLM pricing
))
def my_expensive_tool(data: str) -> dict: ...
```

---

## Configuration Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `max_cost_per_call` | `float \| None` | `None` | Max cost per single call. `None` = unlimited |
| `max_cost_per_session` | `float \| None` | `None` | Max cumulative cost per session. `None` = unlimited |
| `max_calls_per_session` | `int \| None` | `None` | Max call count per session. `None` = unlimited |
| `alert_threshold` | `float` | `0.80` | Fraction of limit at which to emit a warning (0–1) |
| `on_exceed` | `GuardAction` | `BLOCK` | Action when limit exceeded: `BLOCK`, `WARN`, or `LOG` |
| `cost_per_call` | `float \| None` | `None` | Explicit fixed fallback cost when dynamic pricing cannot produce a known price |
| `use_dynamic_llm_costs` | `bool` | `True` | Enable provider-response-based LLM cost tracking |
| `model_pricing_overrides` | `dict[str, tuple[float, float]] \| None` | `None` | Per-model input/output pricing overrides in dollars per 1M tokens |
| `record_llm_spend` | `bool` | `True` | Emit LLM spend metadata into traces, telemetry, and reports |
| `cost_ledger` | `CostLedger \| None` | `None` | Optional ledger for persisting spend events beyond in-memory session accounting |

---

## Real LLM Cost Tracking

For provider-backed LLM calls, prefer wrapping the client instead of manually mutating `result.cost` in an `after_call` hook. The wrapper reads the provider's `usage` payload, resolves pricing, records spend once, and returns the native SDK response unchanged. This gives you a cleaner story in production: budget control for model calls, plus guarded execution for tool calls, in the same library.

```python
import os
from openai import OpenAI
from agentguard import InMemoryCostLedger, TokenBudget
from agentguard.integrations import guard_openai_client

budget = TokenBudget(
    max_cost_per_session=5.00,
    max_calls_per_session=100,
)
ledger = InMemoryCostLedger()
budget.config.cost_ledger = ledger

client = guard_openai_client(
    OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    budget=budget,
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this document"}],
)

print(budget.session_spend)
print(ledger.query(model="gpt-4o"))
```

Pricing resolution order:

1. `model_pricing_overrides`
2. LiteLLM model pricing
3. explicit `cost_per_call`
4. otherwise usage is tracked and cost remains unknown

Install LiteLLM support with:

```bash
pip install awesome-agentguard[costs]
```

---

## Session Management

Budget tracking is per-session. Sessions are identified by `session_id` in the `GuardConfig`:

```python
from agentguard import guard, GuardConfig
from agentguard.core.types import BudgetConfig
import uuid

def create_agent_session() -> str:
    session_id = str(uuid.uuid4())
    return session_id

# Each user gets their own budget
def run_agent_for_user(user_id: str) -> None:
    config = GuardConfig(
        budget=BudgetConfig(
            max_cost_per_session=1.00,
            max_calls_per_session=50,
        ),
        session_id=f"user:{user_id}",
        record=True,
    )
    from agentguard.core.guard import GuardedTool
    search = GuardedTool(search_fn, config=config)
    # This user's calls are tracked independently
    result = search(query="Python tutorials")
```

Without `session_id`, all calls to the same `GuardedTool` instance share a budget counter.

---

## Handling Budget Exceeded

```python
from agentguard.core.types import BudgetExceededError

try:
    result = call_openai("Summarise this 10,000 word document")
except BudgetExceededError as e:
    print(f"Budget exceeded: spent ${e.spent:.4f} of ${e.limit:.4f}")
    return {"error": "Budget exceeded — try a shorter query"}
```

---

## Alert Threshold

The `alert_threshold` triggers a warning *before* the budget is exceeded. This lets you monitor spending without hard-blocking calls:

```python
from agentguard.core.types import BudgetConfig

config = BudgetConfig(
    max_cost_per_session=10.00,
    alert_threshold=0.80,   # Warning at $8.00
)
```

When the alert fires, you'll see a log warning:
```
WARNING: Budget alert: $8.12 of $10.00 session budget used.
```

---

## Common Patterns

### Hard limit with soft alert

```python
BudgetConfig(
    max_cost_per_session=10.00,
    alert_threshold=0.80,           # Warn at $8
    on_exceed=GuardAction.BLOCK,    # Block at $10
)
```

### Soft limit (log and continue)

Useful for auditing without hard enforcement:

```python
BudgetConfig(
    max_cost_per_session=10.00,
    on_exceed=GuardAction.LOG,  # Record but don't block
)
```

### Per-call limit for expensive single calls

```python
BudgetConfig(
    max_cost_per_call=0.50,   # Each call can't cost more than $0.50
    cost_per_call=None,       # Prefer real usage-based pricing
    use_dynamic_llm_costs=True,
)
```

### Daily budget reset

Create a new `GuardedTool` instance each day, or use a `session_id` with a date component:

```python
from datetime import date
from agentguard import GuardConfig
from agentguard.core.types import BudgetConfig

daily_config = GuardConfig(
    budget=BudgetConfig(max_cost_per_session=50.00),
    session_id=f"daily:{date.today().isoformat()}",
)
```

---

## Troubleshooting

### Budget resets unexpectedly

Budgets are tracked in-memory per `GuardedTool` instance. If you restart your process or create a new instance, the budget resets. For persistent budgets across restarts, persist the spend in your own storage and check it in a `before_call` hook:

```python
def check_persistent_budget(call: ToolCall) -> None:
    spent = redis.get(f"budget:{call.session_id}") or 0.0
    if float(spent) >= MAX_BUDGET:
        raise BudgetExceededError(call.tool_name, float(spent), MAX_BUDGET)

@guard(before_call=check_persistent_budget)
def expensive_tool(): ...
```

### `BudgetExceededError` not raised

Check that `on_exceed` is set to `GuardAction.BLOCK`. With `WARN` or `LOG`, the call proceeds.

### Cost not tracked

For tool calls, `cost_per_call` remains the fixed-cost mechanism. For LLM API calls, wrap the provider client with `guard_openai_client`, `guard_anthropic_client`, or `guard_openai_compatible_client`. If the response includes usage but pricing cannot be resolved, agentguard records usage and marks cost as unknown unless you explicitly configured `cost_per_call` as a fallback.
