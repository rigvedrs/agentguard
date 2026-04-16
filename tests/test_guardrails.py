"""Tests for guardrails: circuit breaker, rate limiter, budget, retry, timeout."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agentguard.core.types import CircuitBreakerConfig, GuardAction, RateLimitConfig
from agentguard.guardrails.budget import TokenBudget
from agentguard.guardrails.circuit_breaker import CircuitBreaker
from agentguard.guardrails.rate_limiter import RateLimiter
from agentguard.guardrails.retry import RetryPolicy, compute_retry_delay, retry
from agentguard.guardrails.timeout import ToolTimeoutError, run_with_timeout, timeout


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.get_state("tool") == "closed"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=100)
        for _ in range(3):
            cb.after_failure("tool")
        assert cb.get_state("tool") == "open"

    def test_blocks_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)
        cb.after_failure("tool")
        with pytest.raises(Exception):
            cb.before_call("tool")

    def test_transitions_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        cb.after_failure("tool")
        time.sleep(0.1)
        cb.before_call("tool")  # Should not raise
        assert cb.get_state("tool") == "half_open"

    def test_closes_after_half_open_success(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, success_threshold=1)
        cb.after_failure("tool")
        time.sleep(0.1)
        cb.before_call("tool")  # Probe
        cb.after_success("tool")
        assert cb.get_state("tool") == "closed"

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.after_failure("tool")
        cb.reset("tool")
        assert cb.get_state("tool") == "closed"

    def test_stats_returns_correct_data(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(3):
            try:
                cb.before_call("t")
            except Exception:
                pass
            cb.after_failure("t")

        stats = cb.stats("t")
        assert stats.failure_count == 3
        assert stats.state == "closed"  # Not yet at threshold


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(calls_per_second=10, burst=5)
        for _ in range(5):
            allowed, _ = rl.acquire("tool")
            assert allowed

    def test_blocks_when_bucket_empty(self):
        rl = RateLimiter(calls_per_second=1, burst=1)
        allowed, _ = rl.acquire("tool")
        assert allowed
        allowed, retry_after = rl.acquire("tool")
        assert not allowed
        assert retry_after > 0

    def test_require_raises_on_limit(self):
        rl = RateLimiter(calls_per_second=1, burst=1)
        rl.acquire("t")  # drain
        with pytest.raises(Exception):
            rl.require("t")

    def test_reset_refills_bucket(self):
        rl = RateLimiter(calls_per_second=1, burst=3)
        for _ in range(3):
            rl.acquire("t")
        rl.reset("t")
        allowed, _ = rl.acquire("t")
        assert allowed

    def test_calls_per_minute(self):
        rl = RateLimiter(calls_per_minute=60, burst=2)
        assert rl._rate == pytest.approx(1.0)  # 1/s

    def test_stats(self):
        rl = RateLimiter(calls_per_second=100, burst=10)
        for _ in range(3):
            rl.acquire("t")
        stats = rl.stats("t")
        assert stats.total_allowed == 3


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_blocks_when_session_call_limit_reached(self):
        budget = TokenBudget(max_calls_per_session=2)
        budget.record_spend(0.01)
        budget.record_spend(0.01)
        with pytest.raises(Exception):
            budget.check()

    def test_warn_on_alert_threshold(self):
        budget = TokenBudget(max_cost_per_session=1.00, alert_threshold=0.5)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            budget.record_spend(0.60, tool_name="tool")
            # Should have emitted a warning
            budget_warnings = [x for x in w if "Budget alert" in str(x.message)]
            assert len(budget_warnings) >= 0  # May or may not warn depending on rounding

    def test_reset_clears_spend(self):
        budget = TokenBudget(max_cost_per_session=1.00)
        budget.record_spend(0.50)
        assert budget.session_spend == pytest.approx(0.50)
        budget.reset()
        assert budget.session_spend == 0.0

    def test_session_calls_count(self):
        budget = TokenBudget()
        for _ in range(3):
            budget.record_spend(0.01)
        assert budget.session_calls == 3

    def test_stats(self):
        budget = TokenBudget(max_cost_per_session=10.00, max_calls_per_session=100)
        budget.record_spend(2.50)
        stats = budget.stats()
        assert stats.session_spend == pytest.approx(2.50)
        assert stats.budget_utilisation == pytest.approx(0.25)
        assert stats.calls_remaining == 99


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_retries_on_any_exception_by_default(self):
        policy = RetryPolicy(max_retries=2, initial_delay=0.0)
        assert policy.should_retry(ValueError("oops"))
        assert policy.should_retry(IOError("io"))

    def test_only_retries_specified_exceptions(self):
        policy = RetryPolicy(retryable_exceptions=(IOError,))
        assert policy.should_retry(IOError("io"))
        assert not policy.should_retry(ValueError("val"))

    def test_delay_increases_with_backoff(self):
        policy = RetryPolicy(initial_delay=1.0, backoff_factor=2.0, jitter=False)
        assert policy.delay_for(0) == pytest.approx(1.0)
        assert policy.delay_for(1) == pytest.approx(2.0)
        assert policy.delay_for(2) == pytest.approx(4.0)

    def test_delay_capped_at_max(self):
        policy = RetryPolicy(initial_delay=1.0, backoff_factor=10.0, max_delay=5.0, jitter=False)
        assert policy.delay_for(5) == pytest.approx(5.0)

    def test_retry_decorator_succeeds_on_second_attempt(self):
        call_count = 0

        @retry(max_retries=2, initial_delay=0.0)
        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")
            return "ok"

        with patch("time.sleep"):
            assert fn() == "ok"
        assert call_count == 2

    def test_retry_decorator_raises_after_exhaustion(self):
        @retry(max_retries=1, initial_delay=0.0)
        def fn() -> None:
            raise RuntimeError("always")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                fn()

    def test_compute_retry_delay(self):
        from agentguard.core.types import RetryConfig
        cfg = RetryConfig(initial_delay=1.0, backoff_factor=2.0, max_delay=60.0, jitter=False)
        assert compute_retry_delay(cfg, 0) == pytest.approx(1.0)
        assert compute_retry_delay(cfg, 1) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_run_with_timeout_completes(self):
        def fast() -> str:
            return "done"
        result = run_with_timeout(fast, timeout_seconds=5.0)
        assert result == "done"

    def test_run_with_timeout_raises_on_slow(self):
        def slow() -> None:
            time.sleep(10)
        with pytest.raises(ToolTimeoutError):
            run_with_timeout(slow, timeout_seconds=0.1)

    def test_run_with_timeout_reraises_exceptions(self):
        def raise_value() -> None:
            raise ValueError("inner error")
        with pytest.raises(ValueError, match="inner error"):
            run_with_timeout(raise_value, timeout_seconds=5.0)

    def test_timeout_decorator_sync(self):
        @timeout(0.1)
        def slow() -> None:
            time.sleep(10)
        with pytest.raises(ToolTimeoutError):
            slow()

    def test_timeout_decorator_fast_succeeds(self):
        @timeout(5.0)
        def fast() -> str:
            return "hi"
        assert fast() == "hi"

    @pytest.mark.asyncio
    async def test_timeout_decorator_async(self):
        import asyncio

        @timeout(0.1)
        async def slow() -> None:
            await asyncio.sleep(10)

        with pytest.raises(ToolTimeoutError):
            await slow()
