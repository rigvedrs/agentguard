# Custom Validators

## Overview

agentguard ships with built-in schema and hallucination validators, but every application has its own business rules. Custom validators let you plug arbitrary logic into the guard pipeline.

---

## Anatomy of a Validator

A validator is a callable that receives a `ToolCall` (or a `ToolResult` for output validators) and returns a `ValidationResult`:

```python
from agentguard.core.types import ToolCall, ToolResult, ValidationResult, ValidatorKind

def my_validator(call: ToolCall) -> ValidationResult:
    # Inspect call.kwargs, call.args, call.tool_name, call.session_id...
    
    if something_wrong:
        return ValidationResult(
            valid=False,
            kind=ValidatorKind.CUSTOM,
            message="Human-readable error message",
            details={"key": "extra context for debugging"},
        )
    
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)
```

Attach it to `@guard`:

```python
from agentguard import guard

@guard(custom_validators=[my_validator])
def my_tool(query: str) -> str: ...
```

---

## Writing Input Validators

### Block dangerous SQL

```python
from agentguard.core.types import ToolCall, ValidationResult, ValidatorKind

DANGEROUS_KEYWORDS = {"DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"}

def no_dangerous_sql(call: ToolCall) -> ValidationResult:
    """Block SQL queries with destructive keywords."""
    sql = call.kwargs.get("sql", "")
    found = [kw for kw in DANGEROUS_KEYWORDS if kw in sql.upper()]
    
    if found:
        return ValidationResult(
            valid=False,
            kind=ValidatorKind.CUSTOM,
            message=f"Dangerous SQL keyword(s): {', '.join(found)}",
            details={"keywords": found, "sql": sql[:200]},
        )
    
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

@guard(validate_input=True, custom_validators=[no_dangerous_sql])
def run_query(sql: str) -> list[dict]: ...
```

### Validate URL safety

```python
import re
from agentguard.core.types import ToolCall, ValidationResult, ValidatorKind

BLOCKED_DOMAINS = {"internal.corp.com", "localhost", "127.0.0.1", "169.254.169.254"}

def safe_url_only(call: ToolCall) -> ValidationResult:
    """Prevent the agent from calling internal endpoints."""
    url = call.kwargs.get("url", "")
    match = re.search(r"https?://([^/]+)", url)
    
    if not match:
        return ValidationResult(
            valid=False, kind=ValidatorKind.CUSTOM,
            message=f"Invalid URL format: {url}"
        )
    
    domain = match.group(1).lower()
    if any(domain == blocked or domain.endswith(f".{blocked}")
           for blocked in BLOCKED_DOMAINS):
        return ValidationResult(
            valid=False, kind=ValidatorKind.CUSTOM,
            message=f"Blocked domain: {domain}",
            details={"url": url, "domain": domain},
        )
    
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

@guard(custom_validators=[safe_url_only])
def fetch_url(url: str) -> str: ...
```

### Enforce parameter ranges

```python
from agentguard.core.types import ToolCall, ValidationResult, ValidatorKind

def safe_limit(call: ToolCall) -> ValidationResult:
    """Ensure limit parameter is within safe bounds."""
    limit = call.kwargs.get("limit", 100)
    
    if not isinstance(limit, int):
        return ValidationResult(
            valid=False, kind=ValidatorKind.CUSTOM,
            message=f"limit must be an integer, got {type(limit).__name__}"
        )
    
    if limit > 10_000:
        return ValidationResult(
            valid=False, kind=ValidatorKind.CUSTOM,
            message=f"limit {limit} exceeds maximum of 10,000",
            details={"limit": limit, "max": 10_000},
        )
    
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)
```

---

## Writing Output Validators

Output validators inspect the tool's return value. They receive a `ToolResult`:

```python
from agentguard.core.types import ToolResult, ValidationResult, ValidatorKind

def no_pii_in_output(result: ToolResult) -> ValidationResult:
    """Detect potential PII in tool output."""
    import re
    
    output = str(result.return_value)
    
    # Simple SSN pattern
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', output):
        return ValidationResult(
            valid=False, kind=ValidatorKind.CUSTOM,
            message="Potential SSN detected in output",
        )
    
    # Credit card pattern (basic Luhn check would be better)
    if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', output):
        return ValidationResult(
            valid=False, kind=ValidatorKind.CUSTOM,
            message="Potential credit card number in output",
        )
    
    return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)
```

Attach to `validate_output`:

```python
@guard(validate_output=True, custom_validators=[no_pii_in_output])
def search_user_data(email: str) -> dict: ...
```

!!! note "Input vs Output Validators"
    The `custom_validators` list is passed to both input and output validation phases. The validator must handle both `ToolCall` and `ToolResult` inputs — check `isinstance(call_or_result, ToolResult)` to distinguish them.

---

## Combining Multiple Validators

```python
from agentguard import guard

validators = [
    no_dangerous_sql,
    safe_limit,
    require_authenticated_session,
]

@guard(
    validate_input=True,
    custom_validators=validators,
)
def run_query(sql: str, limit: int = 100) -> list[dict]: ...
```

Validators run in order. The first failure stops the chain and raises `ValidationError`.

---

## Stateful Validators

Validators can carry state (e.g., rate limiting, audit logging):

```python
from agentguard.core.types import ToolCall, ValidationResult, ValidatorKind
import threading

class AuditLogger:
    """Log every tool call to an audit trail."""
    
    def __init__(self, audit_store):
        self.store = audit_store
        self._lock = threading.Lock()
    
    def __call__(self, call: ToolCall) -> ValidationResult:
        with self._lock:
            self.store.append({
                "tool": call.tool_name,
                "session": call.session_id,
                "timestamp": call.timestamp.isoformat(),
                "args": call.kwargs,
            })
        return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

audit_log = []
auditor = AuditLogger(audit_log)

@guard(custom_validators=[auditor])
def sensitive_operation(data: str) -> dict: ...
```

---

## Using `SemanticValidator`

For validation rules that are applied per-tool and registered by name, use `SemanticValidator`:

```python
from agentguard.validators import SemanticValidator

validator = SemanticValidator()

# Register a rule for "run_query"
validator.register(
    "run_query",
    lambda call: (
        ValidationResult(
            valid=False,
            kind=ValidatorKind.CUSTOM,
            message="SQL too long",
        )
        if len(call.kwargs.get("sql", "")) > 5000
        else ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)
    )
)

@guard(custom_validators=[validator])
def run_query(sql: str) -> list[dict]: ...
```

---

## Best Practices

1. **Keep validators pure** — avoid side effects in validation logic. Use `after_call` hooks for side effects like audit logging.

2. **Return detailed error messages** — the agent sees the error and may try to correct its call. Clear messages help: `"limit 50000 exceeds maximum of 10000"` is more actionable than `"invalid limit"`.

3. **Use `details` for debugging** — `ValidationResult.details` is included in trace logs. Put diagnostic information there.

4. **Validate at the right level** — use `validate_input=True` for business rule checks on arguments, and `validate_output=True` for checks on return values. Don't use validators as a replacement for proper type hints.

5. **Test your validators independently** — validators are plain Python callables. Test them directly:

```python
def test_no_dangerous_sql():
    from agentguard.core.types import ToolCall
    
    safe_call = ToolCall(tool_name="run_query", kwargs={"sql": "SELECT * FROM users"})
    result = no_dangerous_sql(safe_call)
    assert result.valid
    
    dangerous_call = ToolCall(tool_name="run_query", kwargs={"sql": "DROP TABLE users"})
    result = no_dangerous_sql(dangerous_call)
    assert not result.valid
    assert "DROP" in result.message
```
