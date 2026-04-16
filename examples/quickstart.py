"""agentguard quickstart example.

Demonstrates the zero-config @guard decorator and basic features.
Run with: python examples/quickstart.py
"""

from __future__ import annotations

import json
import time

# Make sure the src layout is on the path when running directly
from agentguard import (
    CircuitBreaker,
    GuardConfig,
    HallucinationDetector,
    TokenBudget,
    guard,
)
from agentguard.testing import TraceRecorder, assert_tool_call


# ---------------------------------------------------------------------------
# 1. Zero-config usage
# ---------------------------------------------------------------------------


@guard
def fetch_user(user_id: int) -> dict:
    """Fetch a user record by ID."""
    # Simulating a real function
    if user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id}")
    return {"id": user_id, "name": "Alice", "email": "alice@example.com"}


# ---------------------------------------------------------------------------
# 2. Full configuration
# ---------------------------------------------------------------------------


budget = TokenBudget(max_calls_per_session=10, max_cost_per_session=1.00)
cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)


@guard(
    validate_input=True,
    validate_output=True,
    max_retries=2,
    timeout=5.0,
    budget=budget.config,
    circuit_breaker=cb.config,
    record=True,
    trace_dir="/tmp/agentguard_demo",
)
def search_products(query: str, max_results: int = 10) -> list[dict]:
    """Search for products matching the query."""
    time.sleep(0.01)  # Simulate a small network delay
    return [
        {"id": i, "name": f"Product {i}", "relevance": 1.0 / (i + 1)}
        for i in range(min(max_results, 5))
    ]


# ---------------------------------------------------------------------------
# 3. Hallucination detection
# ---------------------------------------------------------------------------


detector = HallucinationDetector()
detector.register_tool(
    "get_stock_price",
    expected_latency_ms=(50, 5000),
    required_fields=["symbol", "price"],
    response_patterns=[r'"price":\s*[\d.]+'],
)


@guard(detect_hallucination=True)
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a symbol."""
    time.sleep(0.05)  # Simulate API call
    return {"symbol": symbol, "price": 150.25, "currency": "USD"}


# ---------------------------------------------------------------------------
# Run the examples
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard Quickstart Demo")
    print("=" * 60)

    # 1. Zero-config
    print("\n1. Zero-config @guard usage")
    user = fetch_user(42)
    print(f"   Fetched user: {user['name']}")

    try:
        fetch_user(-1)
    except RuntimeError as e:
        print(f"   Expected error caught: {e}")

    # 2. Full-config with trace recording
    print("\n2. Full-config with recording")
    with TraceRecorder(storage="/tmp/agentguard_demo") as recorder:
        products = search_products("laptop", max_results=3)
        print(f"   Found {len(products)} products")

        # Verify the recorded trace
        entries = recorder.entries()
        if entries:
            last_entry = entries[-1]
            assert_tool_call(last_entry).succeeded().returned_list().returned_non_empty()
            print(f"   Trace recorded: {last_entry.result.execution_time_ms:.1f}ms")

    stats = recorder.stats()
    print(f"   Session stats: {stats['total_calls']} call(s)")

    # 3. Hallucination detection
    print("\n3. Hallucination detection")
    price_data = get_stock_price("AAPL")
    print(f"   Stock price for AAPL: ${price_data['price']}")

    # 4. Budget usage
    print("\n4. Budget tracking")
    print(f"   Session spend: ${budget.session_spend:.4f}")
    print(f"   Session calls: {budget.session_calls}")

    # 5. Registry
    print("\n5. Registered tools")
    from agentguard.core.registry import global_registry
    for name in global_registry.names():
        reg = global_registry.require(name)
        print(f"   {name}: {reg.call_count} call(s)")

    print("\n✓ Quickstart complete!")


if __name__ == "__main__":
    main()
