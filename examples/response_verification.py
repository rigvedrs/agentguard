"""Response verification example for agentguard.

Demonstrates multi-signal response anomaly detection for tool calls.
The verifier checks whether responses violate expected contracts —
anomalous execution timing, missing required fields, pattern mismatches,
or statistically unusual values.

What this catches:
- Responses arriving in < 2ms (no real I/O: mock left in production,
  cache misconfiguration, test stub not removed)
- Missing required fields (API schema drift, partial responses)
- Pattern mismatches (error body returned as success with status 200)
- Values outside historical norms (unit changes, data corruption)

Run with: python examples/response_verification.py
"""

from __future__ import annotations

import time
from agentguard import guard, GuardConfig, ResponseVerifier
from agentguard import AnomalousResponseError


# ---------------------------------------------------------------------------
# Set up the verifier with tool profiles
# ---------------------------------------------------------------------------


verifier = ResponseVerifier(threshold=0.5)

# Register what NORMAL responses look like for each tool
verifier.register_tool(
    "get_weather",
    expected_latency_ms=(100, 5000),        # Real weather API: 100ms–5s
    required_fields=["temperature", "humidity", "conditions"],
    response_patterns=[r'"temperature":\s*-?\d'],
)

verifier.register_tool(
    "search_web",
    expected_latency_ms=(200, 10000),       # Real search: 200ms–10s
    required_fields=["results", "total"],
)


# ---------------------------------------------------------------------------
# Define guarded tools
# ---------------------------------------------------------------------------


@guard(verify_response=True, record=True, trace_dir="/tmp/agentguard_verification")
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    time.sleep(0.15)  # Simulate real API latency (150ms)
    return {
        "city": city,
        "temperature": 18.5,
        "humidity": 65,
        "conditions": "partly cloudy",
    }

# Register the response profile on the guarded tool
get_weather.register_response_profile(
    expected_latency_ms=(100, 5000),
    required_fields=["temperature", "humidity", "conditions"],
)


@guard(verify_response=True)
def search_web(query: str) -> dict:
    """Search the web."""
    time.sleep(0.25)  # Simulate real search latency
    return {
        "query": query,
        "results": [
            {"title": f"Result for {query}", "url": "https://example.com"},
        ],
        "total": 1,
    }

search_web.register_response_profile(
    expected_latency_ms=(200, 10000),
    required_fields=["results", "total"],
)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_real_call() -> None:
    """Show that normal tool calls pass response verification."""
    print("\n--- Real Tool Call (should pass) ---")
    result = get_weather("London")
    print(f"  Result: {result}")
    print("  ✓ Passed response verification")


def demo_standalone_verifier() -> None:
    """Use the verifier standalone to check suspicious responses."""
    print("\n--- Standalone Verifier ---")

    # Suspicious: sub-ms response time — no real I/O occurred
    fast_result = verifier.verify(
        tool_name="get_weather",
        execution_time_ms=0.3,  # 0.3ms — a real API call is impossible this fast
        response={"temperature": 72, "humidity": 50, "conditions": "sunny"},
    )
    print(f"  Fast response (0.3ms): anomalous={fast_result.is_anomalous}, "
          f"confidence={fast_result.confidence:.2f}")
    print(f"  Signals: {fast_result.signals}")

    # Suspicious: missing required fields (API schema drift)
    incomplete = verifier.verify(
        tool_name="get_weather",
        execution_time_ms=200,
        response={"city": "London"},  # Missing temperature, humidity, conditions
    )
    print(f"  Missing fields: anomalous={incomplete.is_anomalous}, "
          f"confidence={incomplete.confidence:.2f}")

    # Normal response — passes all signals
    normal = verifier.verify(
        tool_name="get_weather",
        execution_time_ms=350,
        response={"temperature": 18, "humidity": 65, "conditions": "rain"},
    )
    print(f"  Normal response: anomalous={normal.is_anomalous}, "
          f"confidence={normal.confidence:.2f}")


def demo_signal_weights() -> None:
    """Show how per-tool weights affect confidence scoring."""
    print("\n--- Per-Tool Signal Weights ---")

    custom = ResponseVerifier(threshold=0.5)

    # Register with latency-heavy weights — useful for tools with
    # very consistent network I/O where timing is a reliable signal
    custom.register_tool(
        "api",
        expected_latency_ms=(50, 3000),
        required_fields=["data", "status"],
        latency_weight=0.7,
        fields_weight=0.2,
        patterns_weight=0.1,
    )

    # Fast but complete — latency signal dominates
    result = custom.verify("api", execution_time_ms=1.0, response={"data": [1, 2], "status": "ok"})
    print(f"  Fast + complete: confidence={result.confidence:.2f} (latency-heavy weighting)")

    # Normal speed but missing fields
    result = custom.verify("api", execution_time_ms=500, response={"error": "not found"})
    print(f"  Slow + incomplete: confidence={result.confidence:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard Response Verification Demo")
    print("=" * 60)

    demo_real_call()
    demo_standalone_verifier()
    demo_signal_weights()

    print("\n✓ Response verification example complete!")


if __name__ == "__main__":
    main()
