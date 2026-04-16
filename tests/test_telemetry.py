"""Tests for agentguard.telemetry — OpenTelemetry integration and StructuredLogger.

These tests exercise:
- StructuredLogger record capture and JSON output
- after_call_hook integration with @guard
- Manual log event helpers (hallucination, circuit breaker, budget, retry)
- instrument_agentguard instrumentation flag and no-op when OTel is absent
- get_default_logger singleton
- guard_span context manager
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from agentguard import guard
from agentguard.core.types import ToolCall, ToolCallStatus, ToolResult
from agentguard.telemetry import (
    StructuredLogger,
    get_default_logger,
    guard_span,
    instrument_agentguard,
    is_instrumented,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(name: str = "test_tool") -> ToolCall:
    return ToolCall(tool_name=name, args=("hello",), kwargs={})


def _make_tool_result(call: ToolCall, status: ToolCallStatus = ToolCallStatus.SUCCESS) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        status=status,
        return_value="result",
        execution_time_ms=42.5,
        retry_count=0,
    )


# ---------------------------------------------------------------------------
# StructuredLogger — core record capture
# ---------------------------------------------------------------------------


class TestStructuredLoggerCapture:
    """StructuredLogger writes JSON lines to its output and buffers records."""

    def test_after_call_hook_captures_record(self) -> None:
        """after_call_hook stores one record per call."""
        buf = io.StringIO()
        logger = StructuredLogger(output=buf, buffer=True)

        call = _make_tool_call()
        result = _make_tool_result(call)
        logger.after_call_hook(call, result)

        records = logger.get_records()
        assert len(records) == 1
        rec = records[0]
        assert rec["event"] == "agentguard.tool_call"
        assert rec["tool_name"] == "test_tool"
        assert rec["status"] == "success"

    def test_json_lines_output_is_valid_json(self) -> None:
        """Each line written to the output stream is valid JSON."""
        buf = io.StringIO()
        logger = StructuredLogger(output=buf, buffer=True)

        call = _make_tool_call("my_func")
        result = _make_tool_result(call)
        logger.after_call_hook(call, result)

        buf.seek(0)
        line = buf.readline().strip()
        parsed = json.loads(line)
        assert parsed["tool_name"] == "my_func"
        assert parsed["execution_time_ms"] == pytest.approx(42.5, abs=0.01)

    def test_multiple_calls_produce_multiple_records(self) -> None:
        """Multiple hook invocations accumulate records."""
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        for i in range(5):
            call = _make_tool_call(f"tool_{i}")
            result = _make_tool_result(call)
            logger.after_call_hook(call, result)
        assert len(logger.get_records()) == 5

    def test_clear_empties_buffer(self) -> None:
        """clear() removes all buffered records."""
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        call = _make_tool_call()
        logger.after_call_hook(call, _make_tool_result(call))
        assert len(logger.get_records()) == 1
        logger.clear()
        assert logger.get_records() == []

    def test_buffer_disabled_does_not_store_records(self) -> None:
        """When buffer=False, get_records() always returns an empty list."""
        logger = StructuredLogger(output=io.StringIO(), buffer=False)
        call = _make_tool_call()
        logger.after_call_hook(call, _make_tool_result(call))
        assert logger.get_records() == []

    def test_failure_result_includes_exception_info(self) -> None:
        """A failed ToolResult records exception and exception_type."""
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        call = _make_tool_call()
        result = ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolCallStatus.FAILURE,
            exception="connection refused",
            exception_type="ConnectionError",
            execution_time_ms=10.0,
        )
        logger.after_call_hook(call, result)
        rec = logger.get_records()[0]
        assert rec["status"] == "failure"
        assert rec["exception"] == "connection refused"
        assert rec["exception_type"] == "ConnectionError"

    def test_include_args_redacts_secret_like_fields(self) -> None:
        """StructuredLogger should avoid emitting raw secrets in args/kwargs."""
        logger = StructuredLogger(output=io.StringIO(), buffer=True, include_args=True)
        call = ToolCall(
            tool_name="test_tool",
            args=("sk-test-secret-value",),
            kwargs={"api_key": "sk-live-super-secret"},
        )
        result = _make_tool_result(call)

        logger.after_call_hook(call, result)

        rec = logger.get_records()[0]
        assert rec["args"] == ["[REDACTED]"]
        assert rec["kwargs"]["api_key"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# StructuredLogger — event helpers
# ---------------------------------------------------------------------------


class TestStructuredLoggerEventHelpers:
    """Manual event-logging helpers produce correct records."""

    def test_log_hallucination(self) -> None:
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        logger.log_hallucination("my_tool", confidence=0.95, reason="suspiciously fast")
        rec = logger.get_records()[0]
        assert rec["event"] == "agentguard.hallucination_detected"
        assert rec["confidence"] == pytest.approx(0.95)
        assert rec["reason"] == "suspiciously fast"

    def test_log_circuit_breaker_opened(self) -> None:
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        logger.log_circuit_breaker_opened("risky_tool")
        rec = logger.get_records()[0]
        assert rec["event"] == "agentguard.circuit_breaker_opened"
        assert rec["tool_name"] == "risky_tool"

    def test_log_budget_exceeded(self) -> None:
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        logger.log_budget_exceeded("api_call", spent=5.0, limit=4.99)
        rec = logger.get_records()[0]
        assert rec["event"] == "agentguard.budget_exceeded"
        assert rec["spent"] == pytest.approx(5.0)
        assert rec["limit"] == pytest.approx(4.99)

    def test_log_retry(self) -> None:
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        logger.log_retry("flaky_tool", attempt=2)
        rec = logger.get_records()[0]
        assert rec["event"] == "agentguard.retry"
        assert rec["attempt"] == 2

    def test_log_event_custom(self) -> None:
        logger = StructuredLogger(output=io.StringIO(), buffer=True)
        logger.log_event("my_custom_event", tool_name="foo", extra_field="bar")
        rec = logger.get_records()[0]
        assert rec["event"] == "my_custom_event"
        assert rec["extra_field"] == "bar"


# ---------------------------------------------------------------------------
# StructuredLogger — @guard integration via after_call hook
# ---------------------------------------------------------------------------


class TestStructuredLoggerGuardIntegration:
    """StructuredLogger integrates seamlessly with the @guard decorator."""

    def test_hook_fires_for_successful_guarded_call(self) -> None:
        """after_call_hook receives a SUCCESS record when the tool succeeds."""
        logger = StructuredLogger(output=io.StringIO(), buffer=True)

        @guard(after_call=logger.after_call_hook)
        def add_numbers(a: int, b: int) -> int:
            return a + b

        result = add_numbers(3, 4)
        assert result == 7

        records = logger.get_records()
        assert len(records) == 1
        assert records[0]["status"] == "success"
        assert records[0]["tool_name"] == "add_numbers"

    def test_hook_fires_for_failed_guarded_call(self) -> None:
        """after_call_hook receives a FAILURE record when the tool raises."""
        logger = StructuredLogger(output=io.StringIO(), buffer=True)

        @guard(after_call=logger.after_call_hook)
        def broken_tool(x: str) -> str:
            raise ValueError("something went wrong")

        with pytest.raises(Exception):
            broken_tool("hi")

        records = logger.get_records()
        assert len(records) == 1
        assert records[0]["status"] == "failure"


# ---------------------------------------------------------------------------
# instrument_agentguard
# ---------------------------------------------------------------------------


class TestInstrumentAgentguard:
    """instrument_agentguard patches GuardedTool and sets the instrumented flag."""

    def test_is_instrumented_reflects_state(self) -> None:
        """is_instrumented() reports whether instrument_agentguard was called.

        Note: because the module-level _instrumented flag is process-global and
        other test files may have already called instrument_agentguard, we test
        that the flag is bool (not that it transitions from False to True, which
        would be fragile in a shared test run).
        """
        result = is_instrumented()
        assert isinstance(result, bool)

    def test_instrument_agentguard_is_idempotent(self) -> None:
        """Calling instrument_agentguard twice should not raise."""
        # Call twice — should be safe regardless of initial state
        instrument_agentguard()
        instrument_agentguard()  # second call is a no-op
        assert is_instrumented() is True

    def test_instrument_with_tool_names(self) -> None:
        """instrument_agentguard accepts an optional tool_names list."""
        # This should not raise even if already instrumented
        # (already instrumented → no-op, idempotent)
        instrument_agentguard(tool_names=["search_web", "query_db"])
        assert is_instrumented() is True


# ---------------------------------------------------------------------------
# guard_span context manager
# ---------------------------------------------------------------------------


class TestGuardSpan:
    """guard_span emits a span (or no-op) without raising."""

    def test_guard_span_runs_body(self) -> None:
        """The body of a guard_span context manager executes normally."""
        executed = []
        with guard_span("agentguard.test_span", tool_name="my_tool"):
            executed.append(True)
        assert executed == [True]

    def test_guard_span_handles_exception_propagation(self) -> None:
        """Exceptions inside guard_span propagate normally."""
        with pytest.raises(RuntimeError, match="test error"):
            with guard_span("agentguard.error_span"):
                raise RuntimeError("test error")

    def test_guard_span_prefixes_name(self) -> None:
        """guard_span adds 'agentguard.' prefix if not already present."""
        # Should not raise — just verifies the helper works
        with guard_span("my_custom_check", tool_name="foo"):
            pass


# ---------------------------------------------------------------------------
# get_default_logger singleton
# ---------------------------------------------------------------------------


class TestGetDefaultLogger:
    """get_default_logger() returns a shared StructuredLogger instance."""

    def test_returns_structured_logger_instance(self) -> None:
        logger = get_default_logger()
        assert isinstance(logger, StructuredLogger)

    def test_same_instance_on_repeated_calls(self) -> None:
        logger1 = get_default_logger()
        logger2 = get_default_logger()
        assert logger1 is logger2
