"""Tests for the three new agentguard features:

1. Middleware Pipeline  (agentguard.middleware)
2. Policy-as-Code       (agentguard.policy)
3. Multi-Agent Shared State (agentguard.shared)

Each feature has at least 5 tests.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helper to run coroutines in tests
# ---------------------------------------------------------------------------


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ===========================================================================
# Feature 1 — Middleware Pipeline
# ===========================================================================

from agentguard.middleware import (
    Middleware,
    MiddlewareChain,
    MiddlewareContext,
    NextFunc,
    logging_middleware,
    metadata_middleware,
    timing_middleware,
)
from agentguard.core.types import GuardConfig, ToolCallStatus, ToolResult


def _make_ctx(tool_name: str = "test_tool", **metadata) -> MiddlewareContext:
    """Build a MiddlewareContext for unit tests."""
    return MiddlewareContext(
        tool_name=tool_name,
        args=("hello",),
        kwargs={},
        config=GuardConfig(),
        metadata=dict(metadata),
    )


def _success_terminal(return_value: Any = "ok") -> NextFunc:
    """Build a terminal function that returns a successful ToolResult."""

    async def _terminal(ctx: MiddlewareContext) -> ToolResult:
        return ToolResult(
            call_id=ctx.call_id,
            tool_name=ctx.tool_name,
            status=ToolCallStatus.SUCCESS,
            return_value=return_value,
            execution_time_ms=1.0,
        )

    return _terminal


class TestMiddlewarePipeline:
    """Tests for MiddlewareChain and MiddlewareContext."""

    # ------------------------------------------------------------------
    # Test 1 — basic chain execution order
    # ------------------------------------------------------------------
    def test_middleware_execution_order(self):
        """Middleware should execute in registration order (outer-first)."""
        order: list[str] = []

        async def mw_a(ctx, next):
            order.append("a:before")
            result = await next(ctx)
            order.append("a:after")
            return result

        async def mw_b(ctx, next):
            order.append("b:before")
            result = await next(ctx)
            order.append("b:after")
            return result

        chain = MiddlewareChain()
        chain.use(mw_a)
        chain.use(mw_b)

        ctx = _make_ctx()
        run(chain.run(ctx, _success_terminal()))

        assert order == ["a:before", "b:before", "b:after", "a:after"]

    # ------------------------------------------------------------------
    # Test 2 — middleware can modify metadata
    # ------------------------------------------------------------------
    def test_middleware_can_mutate_context(self):
        """Middleware should be able to write to ctx.metadata."""
        captured: dict = {}

        async def inject_mw(ctx, next):
            ctx.metadata["injected"] = 42
            return await next(ctx)

        async def read_mw(ctx, next):
            captured.update(ctx.metadata)
            return await next(ctx)

        chain = MiddlewareChain()
        chain.use(inject_mw)
        chain.use(read_mw)

        run(chain.run(_make_ctx(), _success_terminal()))
        assert captured.get("injected") == 42

    # ------------------------------------------------------------------
    # Test 3 — sync middleware is promoted to async transparently
    # ------------------------------------------------------------------
    def test_sync_middleware_supported(self):
        """Sync middleware functions should work alongside async ones."""
        log: list[str] = []

        def sync_mw(ctx, next):
            log.append("sync_called")
            # sync middleware returns a coroutine from next(ctx)
            return next(ctx)

        chain = MiddlewareChain()
        chain.use(sync_mw)

        result = run(chain.run(_make_ctx(), _success_terminal("sync_ok")))
        assert result.status == ToolCallStatus.SUCCESS
        assert result.return_value == "sync_ok"
        assert "sync_called" in log

    # ------------------------------------------------------------------
    # Test 4 — short-circuit: middleware can skip calling next
    # ------------------------------------------------------------------
    def test_middleware_can_short_circuit(self):
        """A middleware that doesn't call next should terminate the chain."""

        async def blocking_mw(ctx, next):
            # Never calls next — simulates auth failure
            return ToolResult(
                call_id=ctx.call_id,
                tool_name=ctx.tool_name,
                status=ToolCallStatus.VALIDATION_FAILED,
                exception="blocked by middleware",
                execution_time_ms=0.0,
            )

        async def should_not_run(ctx, next):
            raise AssertionError("This middleware should not have been called")

        chain = MiddlewareChain()
        chain.use(blocking_mw)
        chain.use(should_not_run)

        result = run(chain.run(_make_ctx(), _success_terminal()))
        assert result.status == ToolCallStatus.VALIDATION_FAILED
        assert "blocked" in (result.exception or "")

    # ------------------------------------------------------------------
    # Test 5 — middleware raises propagates correctly
    # ------------------------------------------------------------------
    def test_middleware_exception_propagates(self):
        """Exceptions raised inside middleware should propagate to the caller."""

        async def error_mw(ctx, next):
            raise PermissionError("no api key")

        chain = MiddlewareChain()
        chain.use(error_mw)

        with pytest.raises(PermissionError, match="no api key"):
            run(chain.run(_make_ctx(), _success_terminal()))

    # ------------------------------------------------------------------
    # Test 6 — convenience middleware factories
    # ------------------------------------------------------------------
    def test_metadata_middleware_injects_fields(self):
        """metadata_middleware should inject static keys into ctx.metadata."""
        captured: dict = {}

        async def capture_mw(ctx, next):
            captured.update(ctx.metadata)
            return await next(ctx)

        chain = MiddlewareChain()
        chain.use(metadata_middleware(env="test", version="1.2.3"))
        chain.use(capture_mw)

        run(chain.run(_make_ctx(), _success_terminal()))
        assert captured["env"] == "test"
        assert captured["version"] == "1.2.3"

    # ------------------------------------------------------------------
    # Test 7 — timing middleware records elapsed_ms
    # ------------------------------------------------------------------
    def test_timing_middleware_records_elapsed(self):
        """timing_middleware should set ctx.metadata['elapsed_ms']."""
        ctx = _make_ctx()
        chain = MiddlewareChain()
        chain.use(timing_middleware())

        run(chain.run(ctx, _success_terminal()))
        assert "elapsed_ms" in ctx.metadata
        assert ctx.metadata["elapsed_ms"] >= 0.0

    # ------------------------------------------------------------------
    # Test 8 — MiddlewareContext.elapsed_ms helper
    # ------------------------------------------------------------------
    def test_context_elapsed_ms(self):
        """MiddlewareContext.elapsed_ms should return a non-negative float."""
        ctx = _make_ctx()
        elapsed = ctx.elapsed_ms()
        assert elapsed >= 0.0

    # ------------------------------------------------------------------
    # Test 9 — MiddlewareChain.copy creates independent chains
    # ------------------------------------------------------------------
    def test_chain_copy_is_independent(self):
        """Copying a chain should not share middleware lists."""

        async def mw_a(ctx, next):
            return await next(ctx)

        async def mw_b(ctx, next):
            return await next(ctx)

        original = MiddlewareChain()
        original.use(mw_a)

        copy = original.copy()
        copy.use(mw_b)

        assert len(original) == 1
        assert len(copy) == 2

    # ------------------------------------------------------------------
    # Test 10 — empty chain calls terminal directly
    # ------------------------------------------------------------------
    def test_empty_chain_runs_terminal(self):
        """An empty MiddlewareChain should still call the terminal function."""
        chain = MiddlewareChain()
        result = run(chain.run(_make_ctx(), _success_terminal("direct")))
        assert result.return_value == "direct"
        assert result.status == ToolCallStatus.SUCCESS


# ===========================================================================
# Feature 2 — Policy-as-Code
# ===========================================================================

from agentguard.policy import (
    load_policy,
    apply_policy,
    validate_policy,
    PolicyError,
    PolicyValidationError,
    policy_summary,
)
from agentguard.core.types import GuardConfig


_VALID_YAML = textwrap.dedent("""\
    version: "1"
    defaults:
      validate_input: true
      max_retries: 2
      record: false
      trace_dir: "./traces"

    tools:
      search_web:
        timeout: 10.0
        rate_limit:
          calls_per_minute: 60
          burst: 10

      query_database:
        timeout: 30.0
        budget:
          max_cost_per_session: 1.00
          cost_per_call: 0.05
        circuit_breaker:
          failure_threshold: 5
          recovery_timeout: 60
""")

_INVALID_YAML = textwrap.dedent("""\
    version: "1"
    defaults:
      unknown_key: true   # should trigger an error

    tools:
      my_tool:
        timeout: "not_a_number"
""")

_VALID_TOML = textwrap.dedent("""\
    version = "1"

    [defaults]
    validate_input = true
    max_retries = 1

    [tools.search_web]
    timeout = 5.0

    [tools.search_web.rate_limit]
    calls_per_minute = 30
    burst = 5
""")


class TestPolicyAsCode:
    """Tests for load_policy, apply_policy, validate_policy."""

    # ------------------------------------------------------------------
    # Test 1 — load a valid YAML policy
    # ------------------------------------------------------------------
    def test_load_valid_yaml_policy(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)
        assert "search_web" in policy
        assert "query_database" in policy

    # ------------------------------------------------------------------
    # Test 2 — defaults are merged into per-tool configs
    # ------------------------------------------------------------------
    def test_defaults_merged_into_tool_config(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)
        # defaults has validate_input=true and max_retries=2
        assert policy["search_web"].validate_input is True
        assert policy["search_web"].max_retries == 2

    # ------------------------------------------------------------------
    # Test 3 — per-tool settings are applied correctly
    # ------------------------------------------------------------------
    def test_per_tool_settings_applied(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)
        cfg = policy["search_web"]
        assert cfg.timeout == 10.0
        assert cfg.rate_limit is not None
        assert cfg.rate_limit.calls_per_minute == 60
        assert cfg.rate_limit.burst == 10

    # ------------------------------------------------------------------
    # Test 4 — budget config parsed correctly
    # ------------------------------------------------------------------
    def test_budget_config_parsed(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)
        db_cfg = policy["query_database"]
        assert db_cfg.budget is not None
        assert db_cfg.budget.max_cost_per_session == 1.00
        assert db_cfg.budget.cost_per_call == 0.05

    # ------------------------------------------------------------------
    # Test 5 — circuit breaker config parsed correctly
    # ------------------------------------------------------------------
    def test_circuit_breaker_config_parsed(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)
        db_cfg = policy["query_database"]
        assert db_cfg.circuit_breaker is not None
        assert db_cfg.circuit_breaker.failure_threshold == 5
        assert db_cfg.circuit_breaker.recovery_timeout == 60

    # ------------------------------------------------------------------
    # Test 6 — validate returns empty list for valid policy
    # ------------------------------------------------------------------
    def test_validate_valid_policy(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        errors = validate_policy(path)
        assert errors == []

    # ------------------------------------------------------------------
    # Test 7 — validate returns errors for invalid policy
    # ------------------------------------------------------------------
    def test_validate_invalid_policy(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_INVALID_YAML)

        errors = validate_policy(path)
        assert len(errors) > 0
        # Should flag the timeout not being a number
        combined = " ".join(errors)
        assert "timeout" in combined.lower() or "unknown" in combined.lower()

    # ------------------------------------------------------------------
    # Test 8 — load_policy raises PolicyValidationError for invalid file
    # ------------------------------------------------------------------
    def test_load_policy_raises_on_invalid_file(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(_INVALID_YAML)

        with pytest.raises(PolicyValidationError) as exc_info:
            load_policy(path)
        assert exc_info.value.errors

    # ------------------------------------------------------------------
    # Test 9 — load_policy raises FileNotFoundError for missing file
    # ------------------------------------------------------------------
    def test_load_policy_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_policy(tmp_path / "nonexistent.yaml")

    # ------------------------------------------------------------------
    # Test 10 — TOML policy file is loaded correctly
    # ------------------------------------------------------------------
    def test_load_toml_policy(self, tmp_path):
        path = tmp_path / "agentguard.toml"
        path.write_text(_VALID_TOML)

        policy = load_policy(path)
        assert "search_web" in policy
        cfg = policy["search_web"]
        assert cfg.timeout == 5.0
        assert cfg.validate_input is True  # from defaults
        assert cfg.rate_limit is not None
        assert cfg.rate_limit.calls_per_minute == 30

    # ------------------------------------------------------------------
    # Test 11 — apply_policy wraps matching tools
    # ------------------------------------------------------------------
    def test_apply_policy_wraps_tools(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)

        def search_web(q: str) -> dict:
            return {"results": []}

        guarded = apply_policy(policy, [search_web])
        assert "search_web" in guarded

        from agentguard.core.guard import GuardedTool
        assert isinstance(guarded["search_web"], GuardedTool)

    # ------------------------------------------------------------------
    # Test 12 — apply_policy with missing_ok=False raises for unknown tool
    # ------------------------------------------------------------------
    def test_apply_policy_missing_ok_false_raises(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)

        def unlisted_tool(x: str) -> str:
            return x

        with pytest.raises(PolicyError, match="no entry in the policy"):
            apply_policy(policy, [unlisted_tool], missing_ok=False)

    # ------------------------------------------------------------------
    # Test 13 — policy_summary produces non-empty string
    # ------------------------------------------------------------------
    def test_policy_summary(self, tmp_path):
        path = tmp_path / "agentguard.yaml"
        path.write_text(_VALID_YAML)

        policy = load_policy(path)
        summary = policy_summary(policy, file_path=str(path))
        assert "search_web" in summary
        assert "query_database" in summary
        assert "timeout" in summary.lower()

    # ------------------------------------------------------------------
    # Test 14 — unsupported file extension raises PolicyError
    # ------------------------------------------------------------------
    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        with pytest.raises(PolicyError, match="Unsupported"):
            load_policy(path)


# ===========================================================================
# Feature 3 — Multi-Agent Shared State
# ===========================================================================

from agentguard.shared import (
    SharedBudget,
    SharedCircuitBreaker,
    SharedBudgetStats,
    SharedCircuitStats,
    get_shared_budget,
    get_shared_circuit_breaker,
    clear_shared_registry,
)
from agentguard.core.types import CircuitState, BudgetExceededError, CircuitOpenError


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure each test starts with a clean shared registry."""
    clear_shared_registry()
    yield
    clear_shared_registry()


class TestSharedBudget:
    """Tests for SharedBudget."""

    # ------------------------------------------------------------------
    # Test 1 — SharedBudget is registered and retrievable
    # ------------------------------------------------------------------
    def test_shared_budget_registered(self):
        budget = SharedBudget(max_cost_per_session=5.00, shared_id="my-budget")
        retrieved = get_shared_budget("my-budget")
        assert retrieved is budget

    # ------------------------------------------------------------------
    # Test 2 — multiple tools share the same spend counter
    # ------------------------------------------------------------------
    def test_shared_spend_across_tools(self):
        from agentguard import guard
        from agentguard.core.types import BudgetExceededError

        budget = SharedBudget(
            max_cost_per_session=1.00,
            cost_per_call=0.30,
            shared_id="multi-agent-budget",
        )

        @guard(budget=budget.config)
        def tool_a(x: str) -> str:
            return x

        @guard(budget=budget.config)
        def tool_b(x: str) -> str:
            return x

        tool_a("hello")
        tool_b("world")

        # Both calls should be counted against the shared budget
        assert budget.session_calls == 2

    # ------------------------------------------------------------------
    # Test 3 — budget exceeded blocks all tools sharing it
    # ------------------------------------------------------------------
    def test_budget_exceeded_blocks_all_tools(self):
        from agentguard import guard
        from agentguard.core.types import BudgetExceededError

        budget = SharedBudget(
            max_calls_per_session=1,
            shared_id="tight-budget",
        )

        @guard(budget=budget.config)
        def tool_a(x: str) -> str:
            return x

        @guard(budget=budget.config)
        def tool_b(x: str) -> str:
            return x

        # First call succeeds
        tool_a("first")

        # Second call (from either tool) should be blocked
        with pytest.raises(BudgetExceededError):
            tool_b("second")

    # ------------------------------------------------------------------
    # Test 4 — reset clears spend and call count
    # ------------------------------------------------------------------
    def test_reset_clears_state(self):
        budget = SharedBudget(max_cost_per_session=5.00, shared_id="reset-budget")
        budget.record_call()
        budget.record_spend(1.00)

        assert budget.session_calls == 1
        assert budget.session_spend == 1.00

        budget.reset()
        assert budget.session_calls == 0
        assert budget.session_spend == 0.0

    # ------------------------------------------------------------------
    # Test 5 — stats() snapshot is accurate
    # ------------------------------------------------------------------
    def test_stats_snapshot(self):
        budget = SharedBudget(
            max_cost_per_session=10.00,
            max_calls_per_session=5,
            shared_id="stats-budget",
        )
        budget.record_call()
        budget.record_spend(2.50)

        stats = budget.stats()
        assert isinstance(stats, SharedBudgetStats)
        assert stats.session_spend == 2.50
        assert stats.session_calls == 1
        assert stats.max_cost_per_session == 10.00
        assert stats.budget_utilisation == pytest.approx(0.25)
        assert stats.calls_remaining == 4

    # ------------------------------------------------------------------
    # Test 6 — auto-generated shared_id is unique
    # ------------------------------------------------------------------
    def test_auto_shared_id_unique(self):
        b1 = SharedBudget(max_cost_per_session=5.00)
        b2 = SharedBudget(max_cost_per_session=5.00)
        assert b1.shared_id != b2.shared_id

    # ------------------------------------------------------------------
    # Test 7 — registered tools tracked
    # ------------------------------------------------------------------
    def test_registered_tools_tracked(self):
        from agentguard import guard

        budget = SharedBudget(
            max_calls_per_session=100,
            shared_id="tracking-budget",
        )

        @guard(budget=budget.config)
        def alpha(x: str) -> str:
            return x

        @guard(budget=budget.config)
        def beta(x: str) -> str:
            return x

        stats = budget.stats()
        assert "alpha" in stats.registered_tools
        assert "beta" in stats.registered_tools


class TestSharedCircuitBreaker:
    """Tests for SharedCircuitBreaker."""

    # ------------------------------------------------------------------
    # Test 1 — SharedCircuitBreaker is registered and retrievable
    # ------------------------------------------------------------------
    def test_shared_cb_registered(self):
        cb = SharedCircuitBreaker(failure_threshold=3, shared_id="my-cb")
        retrieved = get_shared_circuit_breaker("my-cb")
        assert retrieved is cb

    # ------------------------------------------------------------------
    # Test 2 — failures from multiple tools open the shared circuit
    # ------------------------------------------------------------------
    def test_shared_failures_open_circuit(self):
        from agentguard import guard

        cb = SharedCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=9999,
            shared_id="shared-cb",
        )

        @guard(circuit_breaker=cb.config)
        def flaky_tool(fail: bool) -> str:
            if fail:
                raise RuntimeError("boom")
            return "ok"

        # Accumulate failures from the shared tool
        with pytest.raises(RuntimeError):
            flaky_tool(True)
        with pytest.raises(RuntimeError):
            flaky_tool(True)

        # Circuit should now be open
        assert cb.circuit_state == CircuitState.OPEN

    # ------------------------------------------------------------------
    # Test 3 — open circuit blocks calls from all tools
    # ------------------------------------------------------------------
    def test_open_circuit_blocks_all_tools(self):
        from agentguard import guard
        from agentguard.core.types import CircuitOpenError

        cb = SharedCircuitBreaker(
            failure_threshold=1,
            recovery_timeout=9999,
            shared_id="blocking-cb",
        )

        @guard(circuit_breaker=cb.config)
        def tool_a(x: str) -> str:
            raise RuntimeError("always fails")

        @guard(circuit_breaker=cb.config)
        def tool_b(x: str) -> str:
            return x

        with pytest.raises(RuntimeError):
            tool_a("first")

        # tool_b should also be blocked now
        with pytest.raises(CircuitOpenError):
            tool_b("blocked")

    # ------------------------------------------------------------------
    # Test 4 — reset transitions circuit back to CLOSED
    # ------------------------------------------------------------------
    def test_reset_closes_circuit(self):
        cb = SharedCircuitBreaker(
            failure_threshold=1,
            recovery_timeout=9999,
            shared_id="reset-cb",
        )
        cb.record_failure()
        assert cb.circuit_state == CircuitState.OPEN

        cb.reset()
        assert cb.circuit_state == CircuitState.CLOSED

    # ------------------------------------------------------------------
    # Test 5 — stats() snapshot is accurate
    # ------------------------------------------------------------------
    def test_stats_snapshot(self):
        cb = SharedCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            shared_id="stats-cb",
        )
        cb.record_failure()
        cb.record_failure()

        stats = cb.stats()
        assert isinstance(stats, SharedCircuitStats)
        assert stats.failure_count == 2
        assert stats.state == CircuitState.CLOSED  # threshold not reached yet

    # ------------------------------------------------------------------
    # Test 6 — auto-generated shared_id is unique
    # ------------------------------------------------------------------
    def test_auto_shared_id_unique(self):
        cb1 = SharedCircuitBreaker()
        cb2 = SharedCircuitBreaker()
        assert cb1.shared_id != cb2.shared_id

    # ------------------------------------------------------------------
    # Test 7 — multiple tools share the same failure counter
    # ------------------------------------------------------------------
    def test_multiple_tools_share_failure_counter(self):
        from agentguard import guard

        cb = SharedCircuitBreaker(
            failure_threshold=10,
            recovery_timeout=9999,
            shared_id="counter-cb",
        )

        @guard(circuit_breaker=cb.config)
        def tool_x(fail: bool) -> str:
            if fail:
                raise RuntimeError("err")
            return "ok"

        @guard(circuit_breaker=cb.config)
        def tool_y(fail: bool) -> str:
            if fail:
                raise RuntimeError("err")
            return "ok"

        with pytest.raises(RuntimeError):
            tool_x(True)
        with pytest.raises(RuntimeError):
            tool_y(True)

        stats = cb.stats()
        assert stats.failure_count == 2

    # ------------------------------------------------------------------
    # Test 8 — registered tools are tracked
    # ------------------------------------------------------------------
    def test_registered_tools_tracked(self):
        from agentguard import guard

        cb = SharedCircuitBreaker(
            failure_threshold=10,
            shared_id="tracking-cb",
        )

        @guard(circuit_breaker=cb.config)
        def gamma(x: str) -> str:
            return x

        @guard(circuit_breaker=cb.config)
        def delta(x: str) -> str:
            return x

        stats = cb.stats()
        assert "gamma" in stats.registered_tools
        assert "delta" in stats.registered_tools


# ===========================================================================
# Integration: guard decorator with middleware= kwarg
# ===========================================================================

class TestMiddlewareIntegration:
    """Integration tests for middleware wired into GuardedTool."""

    def test_guard_accepts_middleware_chain(self):
        """guard(middleware=...) should run middleware around execution."""
        from agentguard import guard
        from agentguard.middleware import MiddlewareChain

        log: list[str] = []

        async def tracking_mw(ctx, next):
            log.append(f"before:{ctx.tool_name}")
            result = await next(ctx)
            log.append(f"after:{result.status.value}")
            return result

        chain = MiddlewareChain()
        chain.use(tracking_mw)

        @guard(middleware=chain)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3
        assert log == ["before:add", "after:success"]
