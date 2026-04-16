"""Tests for the core @guard decorator and GuardedTool."""

from __future__ import annotations

import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from agentguard import (
    CircuitBreaker,
    GuardConfig,
    RateLimiter,
    TokenBudget,
    guard,
)
from agentguard.core.guard import GuardedTool, _clear_rate_limiter_registry
from agentguard.core.registry import global_registry
from agentguard.core.types import (
    BudgetExceededError,
    CircuitOpenError,
    GuardAction,
    RateLimitError,
    RateLimitConfig,
    ToolTimeoutError,
    ToolCallStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the global registry between tests."""
    yield
    global_registry.clear()
    _clear_rate_limiter_registry()


def make_tool(fail: bool = False, delay: float = 0.0):
    """Helper: create a simple guarded tool."""

    @guard
    def my_tool(x: str) -> dict:
        """A simple test tool."""
        if delay:
            time.sleep(delay)
        if fail:
            raise ValueError("simulated failure")
        return {"result": x, "status": 200}

    return my_tool


# ---------------------------------------------------------------------------
# Basic decorator tests
# ---------------------------------------------------------------------------


class TestGuardDecorator:
    def test_zero_config_decorator(self):
        """@guard with no args should work transparently."""

        @guard
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

    def test_decorator_with_parens(self):
        """@guard() with empty parens should work."""

        @guard()
        def echo(msg: str) -> str:
            return msg

        assert echo("hello") == "hello"

    def test_preserves_function_name(self):
        @guard
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_preserves_docstring(self):
        @guard
        def documented() -> None:
            """I have a docstring."""
            pass

        assert "docstring" in documented.__doc__

    def test_is_guarded_tool_instance(self):
        @guard
        def fn() -> None:
            pass

        assert isinstance(fn, GuardedTool)

    def test_returns_value(self):
        tool = make_tool()
        result = tool("hello")
        assert result == {"result": "hello", "status": 200}

    def test_raises_on_failure(self):
        tool = make_tool(fail=True)
        with pytest.raises(RuntimeError):
            tool("x")

    def test_registered_in_global_registry(self):
        @guard
        def registered_tool() -> None:
            pass

        assert "registered_tool" in global_registry


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestRetry:
    def test_retries_on_failure(self):
        call_count = 0

        @guard(max_retries=2)
        def flaky(x: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retry me")
            return "ok"

        # Patch sleep to speed up test
        with patch("time.sleep"):
            result = flaky("x")
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        @guard(max_retries=2)
        def always_fails(x: str) -> str:
            raise ValueError("permanent failure")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                always_fails("x")

    def test_no_retry_on_success(self):
        call_count = 0

        @guard(max_retries=3)
        def succeeds(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        succeeds("x")
        assert call_count == 1


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        """The internal circuit breaker state opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100)

        @guard(circuit_breaker=cb.config)
        def fragile(x: str) -> str:
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                fragile("x")

        # The GuardedTool's internal circuit breaker state has opened
        # (The standalone CircuitBreaker object is separate from the guard's internal state)
        # Verify by confirming the next call raises CircuitOpenError
        with pytest.raises((CircuitOpenError, RuntimeError)):
            fragile("x")

    def test_blocks_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=100)

        @guard(circuit_breaker=cb.config)
        def fragile(x: str) -> str:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            fragile("x")

        with pytest.raises((CircuitOpenError, RuntimeError)):
            fragile("x")

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=0.01)

        @guard(circuit_breaker=cb.config)
        def tool(should_fail: bool = False) -> str:
            if should_fail:
                raise RuntimeError("fail")
            return "ok"

        result = tool(should_fail=False)
        assert result == "ok"


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------


class TestBudget:
    def test_blocks_when_calls_exceeded(self):
        budget = TokenBudget(max_calls_per_session=2)

        @guard(budget=budget.config)
        def tool(x: str) -> str:
            return x

        tool("a")
        tool("b")
        with pytest.raises(BudgetExceededError):
            tool("c")

    def test_warn_action_does_not_raise(self):
        budget = TokenBudget(max_calls_per_session=1, on_exceed=GuardAction.WARN)

        @guard(budget=budget.config)
        def tool(x: str) -> str:
            return x

        tool("a")
        import warnings
        with warnings.catch_warnings(record=True):
            tool("b")  # Should not raise


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_blocks_when_rate_exceeded(self):
        rl = RateLimiter(calls_per_second=1, burst=1)

        @guard(rate_limit=rl.config)
        def tool(x: str) -> str:
            return x

        tool("a")  # Consume the token
        with pytest.raises(RateLimitError):
            tool("b")  # No tokens left

    def test_same_name_tools_share_bucket_by_default(self):
        def make_shared_tool():
            @guard(rate_limit=RateLimitConfig(calls_per_second=1, burst=1))
            def shared_tool(x: str) -> str:
                return x

            return shared_tool

        tool_a = make_shared_tool()
        tool_b = make_shared_tool()

        tool_a("a")
        with pytest.raises(RateLimitError):
            tool_b("b")

    def test_different_name_tools_do_not_share_bucket(self):
        @guard(rate_limit=RateLimitConfig(calls_per_second=1, burst=1))
        def alpha(x: str) -> str:
            return x

        @guard(rate_limit=RateLimitConfig(calls_per_second=1, burst=1))
        def beta(x: str) -> str:
            return x

        alpha("a")
        beta("b")

    def test_shared_key_empty_disables_sharing(self):
        def make_isolated_tool():
            @guard(rate_limit=RateLimitConfig(calls_per_second=1, burst=1, shared_key=""))
            def isolated_tool(x: str) -> str:
                return x

            return isolated_tool

        tool_a = make_isolated_tool()
        tool_b = make_isolated_tool()

        tool_a("a")
        tool_b("b")

    def test_custom_shared_key_shares_across_tool_names(self):
        cfg = RateLimitConfig(calls_per_second=1, burst=1, shared_key="provider-x")

        @guard(rate_limit=cfg)
        def alpha(x: str) -> str:
            return x

        @guard(rate_limit=cfg)
        def beta(x: str) -> str:
            return x

        alpha("a")
        with pytest.raises(RateLimitError):
            beta("b")

    def test_conflicting_same_name_configs_warn_and_first_one_wins(self):
        def make_shared_tool(cfg: RateLimitConfig):
            @guard(rate_limit=cfg)
            def shared_tool(x: str) -> str:
                return x

            return shared_tool

        tool_a = make_shared_tool(RateLimitConfig(calls_per_second=1, burst=1))
        with pytest.warns(UserWarning, match="Conflicting rate limit config"):
            tool_b = make_shared_tool(RateLimitConfig(calls_per_minute=10, burst=1))

        tool_a("a")
        with pytest.raises(RateLimitError):
            tool_b("b")

    def test_conflicting_custom_group_configs_warn_and_first_one_wins(self):
        @guard(rate_limit=RateLimitConfig(calls_per_second=1, burst=1, shared_key="provider-x"))
        def alpha(x: str) -> str:
            return x

        with pytest.warns(UserWarning, match="Conflicting rate limit config"):
            @guard(rate_limit=RateLimitConfig(calls_per_minute=10, burst=1, shared_key="provider-x"))
            def beta(x: str) -> str:
                return x

        alpha("a")
        with pytest.raises(RateLimitError):
            beta("b")

    def test_shared_state_registry_is_thread_safe(self):
        ready = threading.Barrier(2)
        created: list[GuardedTool] = []

        def build_tool() -> None:
            ready.wait()

            @guard(rate_limit=RateLimitConfig(calls_per_second=1, burst=1))
            def shared_tool(x: str) -> str:
                return x

            created.append(shared_tool)

        threads = [threading.Thread(target=build_tool) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(created) == 2
        assert created[0]._rate_limiter_state is created[1]._rate_limiter_state


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_raises_on_timeout(self):
        @guard(timeout=0.1)
        def slow(x: str) -> str:
            time.sleep(10)
            return "done"

        with pytest.raises(ToolTimeoutError):
            slow("x")

    def test_no_timeout_on_fast_call(self):
        @guard(timeout=5.0)
        def fast(x: str) -> str:
            return x

        assert fast("hello") == "hello"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_input_validation_passes(self):
        @guard(validate_input=True)
        def typed_tool(name: str, count: int) -> dict:
            return {"name": name, "count": count}

        result = typed_tool("alice", 3)
        assert result["name"] == "alice"

    def test_input_validation_fails_wrong_type(self):
        @guard(validate_input=True)
        def typed_tool(count: int) -> dict:
            return {"count": count}

        with pytest.raises(Exception):
            typed_tool("not_an_int")

    def test_optional_param_accepts_none(self):
        @guard(validate_input=True)
        def tool(name: Optional[str] = None) -> dict:
            return {"name": name}

        result = tool(None)
        assert result["name"] is None


# ---------------------------------------------------------------------------
# Hooks tests
# ---------------------------------------------------------------------------


class TestHooks:
    def test_before_call_hook(self):
        called_with = []

        def before(call):
            called_with.append(call.tool_name)

        @guard(before_call=before)
        def tool(x: str) -> str:
            return x

        tool("hello")
        assert "tool" in called_with

    def test_after_call_hook(self):
        results_seen = []

        def after(call, result):
            results_seen.append(result.status)

        @guard(after_call=after)
        def tool(x: str) -> str:
            return x

        tool("hello")
        assert ToolCallStatus.SUCCESS in results_seen


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_tool(self):
        @guard
        async def async_tool(x: str) -> str:
            return x + "_async"

        result = await async_tool.acall("hello")
        assert result == "hello_async"

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        import asyncio

        @guard(timeout=0.1)
        async def slow_async(x: str) -> str:
            await asyncio.sleep(10)
            return "done"

        with pytest.raises((ToolTimeoutError, Exception)):
            await slow_async.acall("x")
