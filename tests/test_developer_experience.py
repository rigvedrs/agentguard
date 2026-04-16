"""Tests for DX improvements: custom validators, hallucination profiles, budget cost check."""

from __future__ import annotations

import time

import pytest

from agentguard import guard, GuardConfig, CustomValidator
from agentguard.core.registry import global_registry
from agentguard.core.types import (
    BudgetConfig,
    BudgetExceededError,
    GuardAction,
    ToolCall,
    ToolCallStatus,
    ValidationResult,
    ValidatorKind,
)
from agentguard.validators.custom import validator_fn


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    global_registry.clear()


# ---------------------------------------------------------------------------
# Custom validators via @guard decorator
# ---------------------------------------------------------------------------


class TestCustomValidatorsThroughGuard:
    """Test that custom validators run when passed to @guard(custom_validators=[...])."""

    def test_custom_validator_blocks_on_failure(self):
        """Custom validator that fails should block the call."""

        @validator_fn(name="always_fail")
        def always_fail(call, result=None):
            return ValidationResult(
                valid=False,
                kind=ValidatorKind.CUSTOM,
                message="always fails",
            )

        @guard(custom_validators=[always_fail])
        def my_tool(x: str) -> str:
            return x.upper()

        with pytest.raises(RuntimeError, match="always fails"):
            my_tool("hello")

    def test_custom_validator_passes(self):
        """Custom validator that passes should not block."""

        @validator_fn(name="always_pass")
        def always_pass(call, result=None):
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        @guard(custom_validators=[always_pass])
        def my_tool(x: str) -> str:
            return x.upper()

        assert my_tool("hello") == "HELLO"

    def test_add_validator_method(self):
        """Test the add_validator() method on GuardedTool."""

        @validator_fn(name="check_result")
        def check_result(call, result=None):
            if result and isinstance(result, str) and "BAD" in result:
                return ValidationResult(
                    valid=False,
                    kind=ValidatorKind.CUSTOM,
                    message="Result contains BAD",
                )
            return ValidationResult(valid=True, kind=ValidatorKind.CUSTOM)

        @guard
        def my_tool(x: str) -> str:
            return x

        my_tool.add_validator(check_result)
        assert my_tool("GOOD") == "GOOD"

        with pytest.raises(RuntimeError, match="contains BAD"):
            my_tool("BAD value")


# ---------------------------------------------------------------------------
# Hallucination detection persistence
# ---------------------------------------------------------------------------


class TestHallucinationProfilePersistence:
    """Test that hallucination profiles persist across calls."""

    def test_register_profile(self):
        """register_hallucination_profile should not error."""

        @guard(detect_hallucination=True)
        def my_tool(x: str) -> dict:
            return {"data": x}

        result = my_tool.register_hallucination_profile(
            expected_latency_ms=(50, 5000),
            required_fields=["data"],
        )
        # Should return self for chaining
        assert result is my_tool

    def test_detector_persists(self):
        """The detector should persist and have profiles across calls."""

        @guard(detect_hallucination=True)
        def my_tool(x: str) -> dict:
            time.sleep(0.01)  # Simulate real I/O so hallucination check passes
            return {"data": x}

        my_tool.register_hallucination_profile(
            expected_latency_ms=(5, 5000),
            required_fields=["data"],
        )

        # The internal detector should know about the tool
        detector = my_tool._hallucination_detector
        assert my_tool._name in detector._profiles
        # Call should succeed (real execution latency, correct fields)
        result = my_tool("hello")
        assert result == {"data": "hello"}


# ---------------------------------------------------------------------------
# Budget cost pre-check
# ---------------------------------------------------------------------------


class TestBudgetCostPreCheck:
    """Test that budget checks cost limits before execution."""

    def test_cost_budget_blocks_when_exceeded(self):
        """Cost budget should block once session spend exceeds max_cost_per_session."""
        budget_cfg = BudgetConfig(
            max_cost_per_session=0.10,
            on_exceed=GuardAction.BLOCK,
            cost_per_call=0.06,
        )

        @guard(budget=budget_cfg)
        def tool(x: str) -> str:
            return x

        # First call: $0.06 spend recorded after execution — within budget
        tool("a")
        # Second call: pre-check sees $0.06 < $0.10, proceeds, records $0.12 after
        tool("b")
        # Third call: pre-check sees $0.12 > $0.10, blocks
        with pytest.raises(BudgetExceededError, match="cost budget"):
            tool("c")

    def test_call_count_still_works(self):
        """max_calls_per_session should still work alongside cost check."""
        budget_cfg = BudgetConfig(
            max_calls_per_session=2,
            on_exceed=GuardAction.BLOCK,
        )

        @guard(budget=budget_cfg)
        def tool(x: str) -> str:
            return x

        tool("a")
        tool("b")
        with pytest.raises(BudgetExceededError, match="calls per session"):
            tool("c")


# ---------------------------------------------------------------------------
# Async retry config
# ---------------------------------------------------------------------------


class TestAsyncRetryConfig:
    """Test that async path properly uses max_retries kwarg."""

    @pytest.mark.asyncio
    async def test_async_max_retries(self):
        """Async calls should retry when max_retries is set via kwarg."""
        call_count = 0

        @guard(max_retries=2)
        async def flaky_tool(x: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary failure")
            return x

        result = await flaky_tool.acall("hello")
        assert result == "hello"
        assert call_count == 3  # 1 initial + 2 retries
