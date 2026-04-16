# Response Verification

## What It Actually Checks

When a tool runs, it produces a response. Response verification checks whether that response looks *normal* based on rules you define — catching integration bugs, schema drift, and misconfigured test stubs before they silently corrupt your agent's reasoning.

Specifically, it flags responses that:

- Arrived impossibly fast (< 2ms) — indicating no real I/O happened (e.g. a mock left in production, a cache misconfiguration)
- Are missing fields you declared as required — indicating API schema changes or incomplete responses
- Don't match patterns you expect in valid responses — indicating error payloads being returned as success
- Contain statistically anomalous numeric values — indicating unit changes, corrupted data, or unexpected API changes

> **What it does not do:** It cannot detect whether an LLM *fabricated* a fact in its text response. It only inspects the actual return value of your Python function.

---

## How It Works

The verifier combines up to four independent signals into a weighted confidence score:

### Signal 1: Execution Time

Real I/O takes time. A tool that wraps a network call, database query, or file operation will never legitimately complete in under 2ms. If it does, something structural is wrong — a stub wasn't removed, a cache is short-circuiting the real call, or the function never reached the external system.

```
Timing thresholds:
  < 2ms    → almost certainly no I/O occurred (score: high)
  < min_ms → below expected range (score: moderate)
  ≥ min_ms → within expected range (score: 0)
  > 3σ     → statistically anomalous vs historical baseline (score: moderate)
```

You define expected latency per tool:

```python
verifier.register_tool(
    "get_weather",
    expected_latency_ms=(200, 5000),  # weather APIs: 200ms–5s
)
```

### Signal 2: Required Fields

Declare which keys must appear in a valid response. Missing any of them flags the response.

```python
verifier.register_tool(
    "get_weather",
    required_fields=["temperature", "humidity", "conditions"],
)
```

If `humidity` is missing from the response, that's a signal — whether because of an API change, a partial response, or an error body returned with a 200 status.

### Signal 3: Pattern Matching

Register regex patterns that valid responses must match. At least one must match, or the signal fires.

```python
verifier.register_tool(
    "get_weather",
    response_patterns=[
        r'"temperature":\s*-?\d+(\.\d+)?',  # temperature must be a number
        r'"conditions":\s*"[^"]+"',           # conditions must be a string
    ],
)
```

### Signal 4: Statistical Plausibility

Once the verifier has accumulated enough historical values for a numeric field (≥ 5 samples), it flags values that fall outside mean ± 3σ.

```
historical temperature readings: [18, 21, 19, 17, 22, 20]
incoming value: 950
→ flagged as anomalous (outside mean ± 3σ)
```

### Confidence Scoring

The four signals are combined with configurable weights into a single confidence score (0 = looks normal, 1 = definitely anomalous):

```
confidence = (
    timing_weight   × timing_score   +
    schema_weight   × schema_score   +
    pattern_weight  × pattern_score  +
    value_weight    × value_score
) / sum(weights)
```

Default weights: timing=0.3, schema=0.3, patterns=0.2, values=0.2.

---

## Basic Usage

### Standalone verifier

```python
from agentguard import ResponseVerifier

verifier = ResponseVerifier(threshold=0.6)

verifier.register_tool(
    "get_weather",
    expected_latency_ms=(100, 5000),
    required_fields=["temperature", "humidity", "conditions"],
    response_patterns=[r'"temperature":\s*-?\d+'],
)

result = verifier.verify(
    tool_name="get_weather",
    execution_time_ms=0.3,
    response={"temperature": 72, "conditions": "sunny"},
)

print(result.is_anomalous)  # True — missing "humidity", sub-ms timing
print(result.confidence)    # 0.87
print(result.reason)        # "Execution time 0.30ms is below the 2ms minimum..."
print(result.signals)       # raw signal values for debugging
```

### With `@guard`

```python
from agentguard import guard

@guard(verify_response=True)
def get_stock_price(ticker: str) -> dict:
    return requests.get(f"https://api.exchange.com/v1/{ticker}").json()

# Register the expected response profile on the guarded function
get_stock_price.register_response_profile(
    expected_latency_ms=(50, 3000),
    required_fields=["ticker", "price", "currency"],
    response_patterns=[r'"price":\s*\d+(\.\d+)?'],
)
```

When a response is flagged as anomalous, `AnomalousResponseError` is raised:

```python
from agentguard import AnomalousResponseError

try:
    result = get_stock_price("AAPL")
except AnomalousResponseError as e:
    print(f"Anomalous response (confidence: {e.result.confidence:.2f})")
    print(f"Reason: {e.result.reason}")
```

---

## Registering Tool Profiles

For accurate detection, define expected characteristics per tool:

```python
from agentguard import ResponseVerifier

verifier = ResponseVerifier(threshold=0.7)

# External API — network latency expected
verifier.register_tool(
    "get_weather",
    expected_latency_ms=(150, 8000),
    required_fields=["temperature", "humidity", "wind_speed"],
    response_patterns=[
        r'"temperature":\s*-?\d+',
        r'"humidity":\s*\d+',
    ],
)

# Database query — fast, structured results
verifier.register_tool(
    "query_users",
    expected_latency_ms=(5, 2000),
    required_fields=["id", "email", "created_at"],
    response_patterns=[
        r'"id":\s*\d+',
        r'"email":\s*"[^@]+@[^"]+\.[^"]+"',
    ],
)

# Local file read — can be very fast, content varies
verifier.register_tool(
    "read_file",
    expected_latency_ms=(0.5, 500),  # filesystem can be sub-ms
    required_fields=[],
    response_patterns=[],
)
```

---

## Threshold Tuning

Three built-in levels:

### Conservative (fewer false positives)

```python
verifier = ResponseVerifier(threshold=0.8)
```

Use when legitimate tool responses vary significantly, or when blocking a real call is worse than letting an anomaly through.

### Balanced (recommended default)

```python
verifier = ResponseVerifier(threshold=0.6)
```

Use for general-purpose agents with mixed tool types.

### Strict (catches more anomalies, more false positives)

```python
verifier = ResponseVerifier(threshold=0.4)
```

Use when tool results feed critical decisions (financial transactions, access control, medical data).

---

## Customising Signal Weights

```python
verifier = ResponseVerifier(
    threshold=0.6,
    weights={
        "timing": 0.5,    # Timing is your most reliable signal
        "schema": 0.3,
        "patterns": 0.1,
        "semantic": 0.1,
    }
)
```

---

## Interpreting Results

```python
result = verifier.verify(
    tool_name="get_weather",
    execution_time_ms=2.1,
    response={"temperature": 72, "humidity": 60},
)

print(result.is_anomalous)   # True / False
print(result.confidence)     # 0.0–1.0
print(result.reason)         # human-readable explanation
print(result.signals)
# {
#   "timing_score": 0.0,
#   "schema_score": 0.33,   # 1 of 3 required fields missing
#   "pattern_score": 0.5,
#   "semantic_score": 0.0,
#   "timing_ms": 2.1,
#   "expected_latency": (150, 8000),
# }
```

---

## Troubleshooting

### Many false positives (real responses flagged as anomalous)

- Widen `expected_latency_ms` — your tool's real latency may vary more than expected.
- Some tools use in-memory caches that return in < 2ms legitimately. Register a looser timing range:
  ```python
  verifier.register_tool("fast_cache_tool", expected_latency_ms=(0.01, 100))
  ```
- Increase the threshold: `ResponseVerifier(threshold=0.8)`.

### Anomalies not being caught

- Lower the threshold: `ResponseVerifier(threshold=0.4)`.
- Add `response_patterns` to catch responses that look structurally wrong.
- Add `required_fields` for fields that always appear in valid responses.

### Consistent false positives on one tool

Override that tool's profile with a looser configuration rather than raising the global threshold.
