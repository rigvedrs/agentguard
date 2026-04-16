"""Budget enforcement example for agentguard.

Demonstrates cost and call count budgets, alerts, and budget-aware
test generation from recorded traces.

Run with: python examples/budget_enforcement.py
"""

from __future__ import annotations

import time
import warnings

from agentguard import (
    GuardConfig,
    HallucinationDetector,
    TokenBudget,
    guard,
)
from agentguard.core.types import BudgetExceededError, GuardAction
from agentguard.reporting.console import ConsoleReporter
from agentguard.reporting.json_report import JsonReporter
from agentguard.testing import TestGenerator, TraceRecorder, assert_tool_call
from agentguard.core.trace import TraceStore


# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------


# Session-level budget: max 5 calls and max $0.50
session_budget = TokenBudget(
    max_calls_per_session=5,
    max_cost_per_session=0.50,
    alert_threshold=0.60,  # Warn at 60%
    on_exceed=GuardAction.BLOCK,
    cost_per_call=0.05,  # Charge $0.05 per call
)

# Per-call budget: max $0.10 per individual call
strict_budget = TokenBudget(
    max_cost_per_call=0.10,
    max_cost_per_session=1.00,
    on_exceed=GuardAction.WARN,
)


# ---------------------------------------------------------------------------
# Guarded tools
# ---------------------------------------------------------------------------


@guard(
    validate_input=True,
    budget=session_budget.config,
    record=True,
    trace_dir="/tmp/agentguard_budget",
    timeout=5.0,
)
def llm_summarise(text: str, max_tokens: int = 100) -> dict:
    """Summarise text using an LLM (simulated, costs $0.05 per call)."""
    time.sleep(0.01)
    words = text.split()
    summary = " ".join(words[:max_tokens])
    cost = len(words) * 0.0001  # Fake cost
    return {
        "summary": summary,
        "input_tokens": len(words),
        "output_tokens": min(max_tokens, len(words)),
        "cost_usd": cost,
    }


@guard(
    validate_input=True,
    budget=strict_budget.config,
    record=True,
    trace_dir="/tmp/agentguard_budget",
)
def embed_text(text: str) -> list[float]:
    """Generate text embeddings (simulated)."""
    time.sleep(0.005)
    return [0.1 * i for i in range(10)]  # Fake 10-dim embedding


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def demo_call_limit() -> None:
    """Show the call limit in action."""
    print("\n--- Call Limit Demo ---")
    print(f"Budget: max {session_budget._cfg.max_calls_per_session} calls")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the software industry.",
        "Python is a versatile programming language.",
        "Cloud computing enables scalable distributed systems.",
        "Quantum computing may revolutionise cryptography.",
    ]

    successes = 0
    blocked = 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i, text in enumerate(texts):
            try:
                result = llm_summarise(text, max_tokens=5)
                # Record the actual cost
                session_budget.record_spend(result["cost_usd"])
                successes += 1
                print(f"  Call {i+1}: OK — '{result['summary'][:40]}...'")
            except BudgetExceededError as e:
                blocked += 1
                print(f"  Call {i+1}: BLOCKED — {e}")
            except RuntimeError as e:
                # Guard raises RuntimeError wrapping the budget error
                blocked += 1
                print(f"  Call {i+1}: BLOCKED — {e}")

    print(f"\n  Successes: {successes}, Blocked: {blocked}")
    print(f"  Session spend: ${session_budget.session_spend:.4f}")
    stats = session_budget.stats()
    print(f"  Budget utilisation: {(stats.budget_utilisation or 0)*100:.0f}%")


def demo_warn_mode() -> None:
    """Show that WARN mode logs but does not raise."""
    print("\n--- Warn Mode Demo ---")

    warn_budget = TokenBudget(
        max_calls_per_session=1,
        on_exceed=GuardAction.WARN,
    )

    @guard(budget=warn_budget.config)
    def tool(x: str) -> str:
        return x.upper()

    print("  Calling once (within limit)...")
    tool("hello")

    print("  Calling again (exceeds limit — will warn, not raise)...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = tool("world")
        if w:
            print(f"  Warning emitted: {w[-1].message}")
    print(f"  Result (still got one): {result!r}")


def demo_trace_reporting() -> None:
    """Show JSON report generation from traces."""
    print("\n--- Trace Reporting ---")

    store = TraceStore("/tmp/agentguard_budget")
    reporter = JsonReporter(store)
    report = reporter.generate(include_anomalies=True)

    print(f"  Total calls: {report['summary'].get('total_calls', 0)}")
    if report["summary"].get("total_calls", 0) > 0:
        print(f"  Success rate: {report['summary']['success_rate']:.0%}")
        if report["summary"]["latency_ms"]:
            print(f"  Avg latency: {report['summary']['latency_ms']['avg']:.1f}ms")

    anomalies = report.get("anomalies", [])
    if anomalies:
        print(f"  Anomalies detected: {len(anomalies)}")
    else:
        print("  No anomalies detected.")


def demo_test_generation() -> None:
    """Generate test cases from recorded traces."""
    print("\n--- Test Generation ---")

    gen = TestGenerator(traces_dir="/tmp/agentguard_budget")
    code = gen.generate_tests(output="/tmp/agentguard_budget/test_generated.py")

    test_count = code.count("def test_")
    print(f"  Generated {test_count} test function(s)")
    print(f"  Output: /tmp/agentguard_budget/test_generated.py")

    # Show a snippet
    lines = code.splitlines()
    for line in lines[:20]:
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("agentguard Budget Enforcement Demo")
    print("=" * 60)

    demo_call_limit()
    demo_warn_mode()
    demo_trace_reporting()
    demo_test_generation()

    # Console reporter
    print("\n--- Console Reporter ---")
    reporter = ConsoleReporter(verbose=True)
    store = TraceStore("/tmp/agentguard_budget")
    entries = store.read_all()
    if entries:
        reporter.print_session_summary(entries)
    else:
        print("  No entries to report yet.")

    print("\n✓ Budget enforcement example complete!")


if __name__ == "__main__":
    main()
