# Policy as Code

## Overview

For large deployments, hardcoding `GuardConfig` in every function becomes unwieldy. agentguard supports loading guard policies from YAML or TOML files — letting you manage guardrail configuration as code, version-controlled and auditable.

---

## YAML Policy File

Create a `guard_policy.yaml` (or `guard_policy.toml`) at your project root:

```yaml
# guard_policy.yaml
default:
  validate_input: true
  max_retries: 3
  timeout: 30.0
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
  rate_limit:
    calls_per_minute: 60

tools:
  query_database:
    validate_input: true
    validate_output: true
    max_retries: 2
    circuit_breaker:
      failure_threshold: 3
      recovery_timeout: 120
    budget:
      max_calls_per_session: 500

  send_email:
    validate_input: true
    rate_limit:
      calls_per_hour: 100
    budget:
      max_cost_per_session: 5.00

  get_weather:
    validate_input: true
    detect_hallucination: true
    max_retries: 3
```

---

## Loading a Policy

```python
from agentguard.core.policy import load_policy

policy = load_policy("guard_policy.yaml")

# Get config for a specific tool (falls back to default)
config = policy.get_config("query_database")

# Apply to a function
from agentguard import guard

@guard(config=policy.get_config("query_database"))
def query_database(sql: str) -> list[dict]:
    return db.execute(sql)
```

---

## TOML Policy File

```toml
# guard_policy.toml

[default]
validate_input = true
max_retries = 3
timeout = 30.0

[default.circuit_breaker]
failure_threshold = 5
recovery_timeout = 60

[default.rate_limit]
calls_per_minute = 60

[tools.query_database]
validate_input = true
validate_output = true

[tools.query_database.circuit_breaker]
failure_threshold = 3
recovery_timeout = 120

[tools.send_email.rate_limit]
calls_per_hour = 100
```

---

## Environment-Specific Policies

Use separate policy files per environment:

```
config/
  guard_policy.dev.yaml
  guard_policy.staging.yaml
  guard_policy.prod.yaml
```

```python
import os
from agentguard.core.policy import load_policy

env = os.getenv("ENV", "dev")
policy = load_policy(f"config/guard_policy.{env}.yaml")
```

---

## Policy Inheritance

Tools that don't have an explicit configuration inherit from `default`. You can also define groups:

```yaml
# guard_policy.yaml
default:
  validate_input: true
  max_retries: 3

groups:
  external_api:
    detect_hallucination: true
    circuit_breaker:
      failure_threshold: 5

tools:
  get_weather:
    group: external_api
    max_retries: 5   # Overrides group default

  search_web:
    group: external_api
```

---

## Applying Policies at Import Time

For large codebases, apply policies automatically using a decorator factory:

```python
# guards.py
from agentguard.core.policy import load_policy
from agentguard import guard as _guard
from functools import wraps
import functools

policy = load_policy("guard_policy.yaml")

def guard(func=None, *, tool_name=None, **kwargs):
    """Apply guard with policy config for this tool."""
    def decorator(fn):
        name = tool_name or fn.__name__
        config = policy.get_config(name)
        return _guard(fn, config=config)
    
    if func is not None:
        return decorator(func)
    return decorator

# tools.py
from .guards import guard  # Our policy-aware guard

@guard
def query_database(sql: str) -> list[dict]:
    return db.execute(sql)

@guard
def send_email(to: str, subject: str, body: str) -> dict:
    return email_service.send(to, subject, body)
```

---

## Validating Policies

Validate your policy file before deployment:

```bash
agentguard policy validate guard_policy.yaml
```

Output:

```
✓ guard_policy.yaml is valid
  default: validate_input=True, max_retries=3, timeout=30.0
  tools: query_database, send_email, get_weather
  Unknown tools (no registered function): send_email
```
