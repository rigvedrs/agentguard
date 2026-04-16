"""Optional OpenTelemetry instrumentation and structured logging for agentguard.

This module provides two complementary observability strategies:

1. **OpenTelemetry tracing** — emits spans and events to any OTel-compatible
   backend (Jaeger, Zipkin, OTLP, etc.).  Requires the ``opentelemetry-api``
   package to be installed; falls back silently to no-ops when it is absent.

2. **StructuredLogger** — a zero-dependency, JSON-lines logger that can be
   attached to any guarded tool via the ``after_call`` hook.

Usage::

    from agentguard.telemetry import instrument_agentguard, StructuredLogger

    # Auto-instrument every guarded tool (call before decorating tools)
    instrument_agentguard()

    # Or target specific tool names
    instrument_agentguard(tool_names=["search_web", "query_db"])

    # Structured JSON logging (no OTel needed)
    logger = StructuredLogger()

    @guard(after_call=logger.after_call_hook)
    def my_tool(x: str) -> str:
        return x.upper()

    # Retrieve captured log records
    records = logger.get_records()
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

from agentguard.core.redaction import sanitize_value

# ---------------------------------------------------------------------------
# Optional OpenTelemetry imports — graceful no-op fallback
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import (
        NonRecordingSpan,
        Span,
        SpanKind,
        Status,
        StatusCode,
        Tracer,
    )
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False
    otel_trace = None  # type: ignore[assignment]

    # Minimal stubs so the rest of the module compiles without otel installed
    class _NoOpSpan:  # type: ignore[too-few-public-methods]
        """No-op span stub used when opentelemetry-api is not installed."""

        def __enter__(self) -> "_NoOpSpan":
            return self

        def __exit__(self, *_: Any) -> None:
            pass

        def set_attribute(self, *_: Any) -> None:  # noqa: ANN002
            pass

        def add_event(self, *_: Any, **__: Any) -> None:  # noqa: ANN002
            pass

        def set_status(self, *_: Any) -> None:  # noqa: ANN002
            pass

        def record_exception(self, *_: Any, **__: Any) -> None:  # noqa: ANN002
            pass

    class _NoOpContextManager:  # type: ignore[too-few-public-methods]
        """No-op context manager returned by no-op tracer spans."""

        def __enter__(self) -> "_NoOpSpan":
            return _NoOpSpan()

        def __exit__(self, *_: Any) -> None:
            pass

    class _NoOpTracer:  # type: ignore[too-few-public-methods]
        """Minimal stub matching the opentelemetry Tracer interface."""

        def start_as_current_span(self, name: str, **kwargs: Any) -> "_NoOpContextManager":
            return _NoOpContextManager()

        def start_span(self, name: str, **kwargs: Any) -> "_NoOpSpan":
            return _NoOpSpan()

    Span = _NoOpSpan  # type: ignore[assignment,misc]
    Tracer = _NoOpTracer  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_instrumented: bool = False
_instrumented_tools: Optional[set[str]] = None  # None = instrument all
_instrumented_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Tracer accessor
# ---------------------------------------------------------------------------


def _get_tracer() -> Any:
    """Return an OTel Tracer instance, or a no-op stub if OTel is unavailable."""
    if _OTEL_AVAILABLE and otel_trace is not None:
        return otel_trace.get_tracer("agentguard", schema_url="https://opentelemetry.io/schemas/1.24.0")
    return _NoOpTracer()  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Public instrumentation entry point
# ---------------------------------------------------------------------------


def instrument_agentguard(
    tool_names: Optional[list[str]] = None,
) -> None:
    """Enable OpenTelemetry instrumentation for guarded tools.

    This function monkey-patches the :class:`~agentguard.core.guard.GuardedTool`
    execution pipeline so that every qualifying tool call emits:

    - A parent span ``agentguard.tool_call`` with standard attributes.
    - Child spans for each guard check phase.
    - Events for significant occurrences (hallucination, circuit breaker, etc.).

    Args:
        tool_names: Restrict instrumentation to these tool names. When *None*
            (the default), every guarded tool call is instrumented.

    Note:
        Call this function *once* at application startup, before your tools are
        invoked.  Calling it multiple times is idempotent.
    """
    global _instrumented, _instrumented_tools

    with _instrumented_lock:
        if _instrumented:
            return
        _instrumented = True
        _instrumented_tools = set(tool_names) if tool_names is not None else None

    _patch_guarded_tool()


def is_instrumented() -> bool:
    """Return True if :func:`instrument_agentguard` has been called."""
    return _instrumented


# ---------------------------------------------------------------------------
# GuardedTool patching
# ---------------------------------------------------------------------------


def _patch_guarded_tool() -> None:
    """Wrap GuardedTool._execute_sync and _execute_async with OTel spans."""
    from agentguard.core.guard import GuardedTool

    original_sync = GuardedTool._execute_sync
    original_async = GuardedTool._execute_async

    def _patched_sync(self: Any, call: Any) -> Any:  # type: ignore[override]
        if not _should_instrument(self._name):
            return original_sync(self, call)
        return _execute_with_span(self, call, original_sync)

    async def _patched_async(self: Any, call: Any) -> Any:  # type: ignore[override]
        if not _should_instrument(self._name):
            return await original_async(self, call)
        return await _execute_with_span_async(self, call, original_async)

    GuardedTool._execute_sync = _patched_sync  # type: ignore[method-assign]
    GuardedTool._execute_async = _patched_async  # type: ignore[method-assign]


def _should_instrument(tool_name: str) -> bool:
    """Return True if *tool_name* should be instrumented."""
    if not _instrumented:
        return False
    if _instrumented_tools is None:
        return True
    return tool_name in _instrumented_tools


# ---------------------------------------------------------------------------
# Span execution wrappers
# ---------------------------------------------------------------------------


def _execute_with_span(self: Any, call: Any, original_fn: Callable[..., Any]) -> Any:
    """Wrap a synchronous _execute_sync call with an OTel parent span."""
    tracer = _get_tracer()
    with tracer.start_as_current_span(
        "agentguard.tool_call",
        kind=SpanKind.INTERNAL if _OTEL_AVAILABLE else None,  # type: ignore[arg-type]
    ) as parent_span:
        # Record guard check child spans
        _emit_guard_check_spans(tracer, self)

        result = original_fn(self, call)

        # Populate parent span attributes from result
        _annotate_tool_span(parent_span, call.tool_name, result)
        return result


async def _execute_with_span_async(
    self: Any,
    call: Any,
    original_fn: Callable[..., Any],
) -> Any:
    """Wrap an asynchronous _execute_async call with an OTel parent span."""
    tracer = _get_tracer()
    with tracer.start_as_current_span(
        "agentguard.tool_call",
        kind=SpanKind.INTERNAL if _OTEL_AVAILABLE else None,  # type: ignore[arg-type]
    ) as parent_span:
        _emit_guard_check_spans(tracer, self)

        result = await original_fn(self, call)

        _annotate_tool_span(parent_span, call.tool_name, result)
        return result


def _emit_guard_check_spans(tracer: Any, guarded_tool: Any) -> None:
    """Emit child spans for each configured guard check phase."""
    cfg = guarded_tool._config

    # Each entry: (condition, span_name)
    checks = [
        (True, "agentguard.guard.input_validation"),
        (True, "agentguard.guard.output_validation"),
        (cfg.detect_hallucination, "agentguard.guard.hallucination_check"),
        (guarded_tool._circuit_breaker_state is not None, "agentguard.guard.circuit_breaker_check"),
        (guarded_tool._rate_limiter_state is not None, "agentguard.guard.rate_limit_check"),
        (guarded_tool._budget_state is not None, "agentguard.guard.budget_check"),
    ]

    for condition, span_name in checks:
        if condition:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agentguard.tool.name", guarded_tool._name)


def _annotate_tool_span(span: Any, tool_name: str, result: Any) -> None:
    """Set attributes and events on the parent tool_call span after execution."""
    span.set_attribute("agentguard.tool.name", tool_name)
    span.set_attribute("agentguard.tool.status", result.status.value)
    span.set_attribute("agentguard.tool.execution_time_ms", result.execution_time_ms)
    span.set_attribute("agentguard.tool.retry_count", result.retry_count)

    # Emit per-event signals
    if result.hallucination and result.hallucination.is_hallucinated:
        span.add_event(
            "agentguard.hallucination_detected",
            attributes={"confidence": result.hallucination.confidence},
        )

    from agentguard.core.types import ToolCallStatus
    if result.status == ToolCallStatus.CIRCUIT_OPEN:
        span.add_event("agentguard.circuit_breaker_opened")

    if result.status == ToolCallStatus.BUDGET_EXCEEDED:
        span.add_event("agentguard.budget_exceeded")

    if result.retry_count > 0:
        span.add_event(
            "agentguard.retry",
            attributes={"attempt": result.retry_count},
        )

    if result.failed and _OTEL_AVAILABLE:
        span.set_status(Status(StatusCode.ERROR, result.exception or ""))


# ---------------------------------------------------------------------------
# Context manager for manual span emission
# ---------------------------------------------------------------------------


@contextmanager
def guard_span(
    name: str,
    tool_name: str = "",
    **attributes: Any,
) -> Generator[Any, None, None]:
    """Context manager that emits a single agentguard-namespaced OTel span.

    Useful for instrumenting code that falls outside the ``@guard`` decorator.

    Args:
        name: Span name (will be prefixed with ``agentguard.`` if not already).
        tool_name: Optional tool name attribute.
        **attributes: Additional span attributes.

    Example::

        with guard_span("agentguard.custom_check", tool_name="my_tool"):
            perform_custom_check()
    """
    tracer = _get_tracer()
    full_name = name if name.startswith("agentguard.") else f"agentguard.{name}"
    with tracer.start_as_current_span(full_name) as span:
        if tool_name:
            span.set_attribute("agentguard.tool.name", tool_name)
        for key, val in attributes.items():
            span.set_attribute(key, val)
        yield span


# ---------------------------------------------------------------------------
# StructuredLogger — zero-dependency JSON-lines logger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Emit structured JSON-lines log records for every guarded tool call.

    ``StructuredLogger`` is a lightweight alternative to OTel for users who
    need machine-readable observability without setting up a tracing backend.

    Each call produces one JSON object written to *output* (defaults to
    ``sys.stdout``) and optionally stored in memory for programmatic access.

    Usage::

        logger = StructuredLogger()

        @guard(after_call=logger.after_call_hook)
        def my_tool(x: str) -> str:
            return x.upper()

        records = logger.get_records()

    Args:
        output: File-like object to write JSON lines to.  Defaults to
            ``sys.stdout``.
        level: Minimum Python logging level at which records are written.
        include_args: When True, serialise call args/kwargs into the log record.
        buffer: When True, keep all records in memory (accessible via
            :meth:`get_records`).
    """

    def __init__(
        self,
        output: Any = None,
        level: int = logging.INFO,
        include_args: bool = False,
        buffer: bool = True,
        redact_fields: tuple[str, ...] = (),
    ) -> None:
        self._output = output or sys.stdout
        self._level = level
        self._include_args = include_args
        self._buffer = buffer
        self._redact_fields = redact_fields
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Hook — attach to @guard(after_call=logger.after_call_hook)
    # ------------------------------------------------------------------

    def after_call_hook(self, call: Any, result: Any) -> None:
        """Hook compatible with ``GuardConfig.after_call``.

        Args:
            call: The :class:`~agentguard.core.types.ToolCall` that was made.
            result: The :class:`~agentguard.core.types.ToolResult` produced.
        """
        record = self._build_record(call, result)
        self._emit(record)

    # ------------------------------------------------------------------
    # Manual logging helpers
    # ------------------------------------------------------------------

    def log_event(self, event: str, tool_name: str = "", **extra: Any) -> None:
        """Emit an arbitrary agentguard event record.

        Args:
            event: Event name (e.g. ``"agentguard.hallucination_detected"``).
            tool_name: Optional tool name.
            **extra: Additional key/value pairs added to the record.
        """
        record: dict[str, Any] = {
            "event": event,
            "tool_name": tool_name,
            "timestamp": time.time(),
        }
        record.update(extra)
        self._emit(record)

    def log_hallucination(self, tool_name: str, confidence: float, reason: str = "") -> None:
        """Emit a hallucination-detected event record.

        Args:
            tool_name: Name of the tool for which hallucination was detected.
            confidence: Confidence score (0–1).
            reason: Human-readable explanation.
        """
        self.log_event(
            "agentguard.hallucination_detected",
            tool_name=tool_name,
            confidence=confidence,
            reason=reason,
        )

    def log_circuit_breaker_opened(self, tool_name: str) -> None:
        """Emit a circuit-breaker-opened event record."""
        self.log_event("agentguard.circuit_breaker_opened", tool_name=tool_name)

    def log_budget_exceeded(self, tool_name: str, spent: float, limit: float) -> None:
        """Emit a budget-exceeded event record."""
        self.log_event(
            "agentguard.budget_exceeded",
            tool_name=tool_name,
            spent=spent,
            limit=limit,
        )

    def log_retry(self, tool_name: str, attempt: int) -> None:
        """Emit a retry event record."""
        self.log_event("agentguard.retry", tool_name=tool_name, attempt=attempt)

    # ------------------------------------------------------------------
    # Buffer access
    # ------------------------------------------------------------------

    def get_records(self) -> list[dict[str, Any]]:
        """Return all buffered log records.

        Returns:
            A copy of the internal record list (safe to mutate).
        """
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        """Clear the in-memory record buffer."""
        with self._lock:
            self._records.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_record(self, call: Any, result: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "agentguard.tool_call",
            "tool_name": call.tool_name,
            "call_id": call.call_id,
            "status": result.status.value,
            "execution_time_ms": round(result.execution_time_ms, 3),
            "retry_count": result.retry_count,
            "timestamp": time.time(),
        }

        if result.exception:
            record["exception"] = result.exception
            if result.exception_type:
                record["exception_type"] = result.exception_type

        if result.hallucination:
            record["hallucination"] = {
                "is_hallucinated": result.hallucination.is_hallucinated,
                "confidence": result.hallucination.confidence,
                "reason": result.hallucination.reason,
            }

        if result.cost is not None:
            record["cost_usd"] = result.cost
        if result.provider:
            record["provider"] = result.provider
        if result.model:
            record["model"] = result.model
        if result.cost_known:
            record["cost_known"] = result.cost_known

        if self._include_args:
            try:
                record["args"] = list(
                    sanitize_value(call.args, extra_fields=self._redact_fields)
                )
                record["kwargs"] = sanitize_value(
                    call.kwargs,
                    extra_fields=self._redact_fields,
                )
            except Exception:  # noqa: BLE001
                pass

        if call.session_id:
            record["session_id"] = call.session_id

        return record

    def _emit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock:
            if self._buffer:
                self._records.append(record)
            try:
                self._output.write(line + "\n")
                if hasattr(self._output, "flush"):
                    self._output.flush()
            except Exception:  # noqa: BLE001
                pass  # Never raise from a logger


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------

_default_logger: Optional[StructuredLogger] = None


def get_default_logger() -> StructuredLogger:
    """Return the module-level default :class:`StructuredLogger` instance.

    Creates one on first call. Useful when you want a shared logger across
    multiple tools without managing the instance yourself.

    Returns:
        The shared :class:`StructuredLogger`.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = StructuredLogger()
    return _default_logger
