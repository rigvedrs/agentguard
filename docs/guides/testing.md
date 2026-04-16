# Testing

## Overview

Testing AI agent tool calls is hard. The inputs are non-deterministic, external services are unreliable, and manually writing assertions for every possible response is tedious.

agentguard provides three testing utilities:

- **`assert_tool_call`** — fluent assertion builder for trace entries
- **`TestGenerator`** — auto-generate pytest tests from production traces
- **`TraceReplayer`** — replay recorded traces to detect regressions

---

## `assert_tool_call` — Fluent Assertions

`assert_tool_call` provides a readable, chainable assertion API for inspecting `TraceEntry` objects:

```python
from agentguard import assert_tool_call
from agentguard import record_session

with record_session("./traces", backend="sqlite") as recorder:
    result = get_weather("San Francisco")

entry = recorder.entries[-1]

assert_tool_call(entry) \
    .succeeded() \
    .within_ms(5000) \
    .returned_dict() \
    .has_keys("temperature", "humidity") \
    .no_retries()
```

### Available assertions

| Method | Description |
|---|---|
| `.succeeded()` | Asserts `status == SUCCESS` |
| `.failed()` | Asserts `status` is a failure state |
| `.within_ms(n)` | Asserts `execution_time_ms <= n` |
| `.returned_dict()` | Asserts return value is a `dict` |
| `.returned_list()` | Asserts return value is a `list` |
| `.returned_str()` | Asserts return value is a `str` |
| `.has_keys(*keys)` | Asserts all `keys` present in the return dict |
| `.no_retries()` | Asserts `retry_count == 0` |
| `.retried(n)` | Asserts `retry_count == n` |
| `.not_hallucinated()` | Asserts hallucination detection passed |
| `.status(s)` | Asserts specific `ToolCallStatus` |

### Full example

```python
import pytest
from agentguard import guard, assert_tool_call, record_session
from agentguard.core.types import TraceEntry

@guard(validate_input=True, max_retries=2, record=True, trace_backend="sqlite", trace_dir="./test_traces")
def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 72, "humidity": 60}

def test_get_weather_response_structure():
    with record_session("./test_traces", backend="sqlite") as recorder:
        get_weather("Tokyo")
    
    entry: TraceEntry = recorder.entries[-1]
    
    assert_tool_call(entry) \
        .succeeded() \
        .within_ms(1000) \
        .returned_dict() \
        .has_keys("city", "temperature", "humidity") \
        .no_retries()
```

---

## `TestGenerator` — Generate Tests from Traces

After running your agent in production (or staging), generate a regression test suite automatically:

```python
from agentguard.testing import TestGenerator

generator = TestGenerator(traces_dir="./production_traces")
generator.generate_tests(output="tests/test_generated.py")
```

### What gets generated

```python
"""Auto-generated test suite from agentguard production traces.
Generated: 2026-04-08T14:30:22Z
Source: ./production_traces
DO NOT EDIT — regenerate with: agentguard generate ./production_traces
"""

def test_get_weather_0():
    """Recorded: get_weather('San Francisco') → success in 312.1ms"""
    result = get_weather("San Francisco")
    assert isinstance(result, dict)
    assert "temperature" in result
    assert "humidity" in result

def test_search_web_0():
    """Recorded: search_web('Python tutorials') → success in 234.1ms"""
    result = search_web("Python tutorials")
    assert isinstance(result, str)
    assert len(result) > 0

def test_query_database_0():
    """Recorded: query_database('SELECT * FROM users LIMIT 5') → success in 12.1ms"""
    result = query_database("SELECT * FROM users LIMIT 5")
    assert isinstance(result, list)
```

### CLI equivalent

```bash
agentguard generate ./traces --output tests/test_generated.py
```

### Customising generated tests

```python
generator = TestGenerator(
    traces_dir="./traces",
    include_tools=["get_weather", "search_web"],  # Only these tools
    exclude_tools=["debug_tool"],                   # Exclude these
    max_tests_per_tool=10,                          # Cap per tool
)
generator.generate_tests("tests/test_generated.py")
```

---

## `TraceReplayer` — Regression Detection

`TraceReplayer` re-runs recorded calls against your current implementation and compares results. This catches regressions after refactoring:

```python
from agentguard.testing import TraceReplayer

replayer = TraceReplayer(traces_dir="./traces")
results = replayer.replay_all(
    tools={
        "get_weather": get_weather,
        "search_web": search_web,
    }
)

for r in results:
    status = "PASS" if r["match"] else "FAIL"
    print(f"[{status}] {r['tool_name']}({r['call_kwargs']})")
    if not r["match"]:
        print(f"  Expected type: {r['expected_type']}")
        print(f"  Got type:      {r['actual_type']}")
```

### What counts as a "match"

By default, the replayer checks:

1. Both old and new calls succeed (or both fail)
2. Return type is the same
3. Dict keys are the same (for dict responses)

The replayer doesn't require exact value equality, since real API responses naturally vary. Override the comparison:

```python
def my_comparator(expected, actual) -> bool:
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected.keys()) == set(actual.keys())
    return type(expected) == type(actual)

results = replayer.replay_all(
    tools={"get_weather": get_weather},
    comparator=my_comparator,
)
```

### Replaying a single session

```python
results = replayer.replay_session(
    session_id="user_abc_session_42",
    tools={"get_weather": get_weather},
)
```

---

## Writing Tests for Guarded Tools

### Testing with mocked tools

```python
from unittest.mock import patch
import pytest
from agentguard import guard, GuardConfig

config = GuardConfig(validate_input=True, max_retries=2)

@guard(config=config)
def get_weather(city: str) -> dict:
    import requests
    return requests.get(f"https://wttr.in/{city}?format=j1").json()

def test_get_weather_success():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {
            "temperature": 72, "humidity": 60, "conditions": "sunny"
        }
        result = get_weather("London")
    
    assert result["temperature"] == 72
    assert result["humidity"] == 60

def test_get_weather_retry():
    """Verify retries are attempted on failure."""
    call_count = 0
    
    def flaky_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Network error")
        mock = patch("requests.Response")()
        mock.json.return_value = {"temperature": 72}
        return mock
    
    with patch("requests.get", side_effect=flaky_request):
        result = get_weather("London")
    
    assert call_count == 3
    assert result["temperature"] == 72
```

### Testing validation

```python
import pytest
from agentguard.core.types import ValidationError

@guard(validate_input=True)
def add(a: int, b: int) -> int:
    return a + b

def test_valid_input():
    assert add(a=2, b=3) == 5

def test_invalid_input():
    with pytest.raises(Exception):
        add(a="two", b=3)  # Should fail validation
```

### Testing circuit breaker

```python
from agentguard import guard
from agentguard.core.types import CircuitBreakerConfig, CircuitOpenError

@guard(circuit_breaker=CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=9999,  # Never recover during tests
))
def unreliable_api(x: int) -> int:
    if x < 0:
        raise ValueError("Negative input")
    return x * 2

def test_circuit_opens_after_threshold():
    for _ in range(2):
        try:
            unreliable_api(-1)
        except ValueError:
            pass
    
    with pytest.raises(CircuitOpenError):
        unreliable_api(1)  # Circuit is now open
```

---

## Integration with pytest

### Using `record_session` in fixtures

```python
# conftest.py
import pytest
from agentguard import record_session

@pytest.fixture
def trace_recorder(tmp_path):
    """Record all tool calls during a test."""
    with record_session(str(tmp_path), backend="sqlite") as recorder:
        yield recorder

# In tests:
def test_agent_flow(trace_recorder):
    run_agent("Summarise today's weather")
    
    # Inspect what the agent called
    tool_names = [e.tool_name for e in trace_recorder.entries]
    assert "get_weather" in tool_names
    assert "summarise_text" in tool_names
```

### Parameterised tests from traces

```python
import pytest
from agentguard.core.trace import TraceStore

store = TraceStore(directory="./test_fixtures", backend="sqlite")
all_entries = store.read_all()

@pytest.mark.parametrize("entry", all_entries, ids=[e.call_id[:8] for e in all_entries])
def test_replay_recorded_call(entry):
    """Replay each recorded call and verify success."""
    tool = get_tool_by_name(entry.tool_name)
    result = tool(**entry.call.kwargs)
    
    # Basic type check — same return type as recorded
    assert type(result) == type(entry.result.return_value)
```

---

## Best Practices

1. **Generate tests from real traffic** — synthetic test data misses the edge cases your agent discovers in production.

2. **Version exported fixtures intentionally** — keep JSONL exports only when you need portable replay fixtures; otherwise prefer SQLite for day-to-day recording.

3. **Run `TraceReplayer` in CI** — add trace replay to your CI pipeline to catch regressions before deployment.

4. **Keep generated tests short-lived** — regenerate them regularly from fresh traces. Old generated tests test old behavior; new ones test current behavior.

5. **Use `assert_tool_call` for structure, not values** — tool responses change; their structure rarely does. Assert that `"temperature" in result` rather than `result["temperature"] == 72`.
