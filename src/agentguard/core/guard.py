"""The @guard decorator — the primary public API of agentguard.

``@guard`` wraps any callable with validation, hallucination detection,
circuit breaking, rate limiting, budget enforcement, retries, timeouts,
and trace recording — in a single decorator.

Usage::

    from agentguard import guard, GuardConfig

    # Zero-config
    @guard
    def fetch_data(url: str) -> dict:
        ...

    # With options
    @guard(validate_input=True, max_retries=3, timeout=30.0, record=True)
    def query_db(sql: str) -> list[dict]:
        ...

    # With full config object
    config = GuardConfig(validate_input=True, validate_output=True, max_retries=2)
    @guard(config=config)
    def call_api(endpoint: str) -> dict:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
import traceback
import uuid
import warnings
from typing import Any, Callable, Optional, TypeVar, cast, overload

from agentguard.core.registry import ToolRegistration, global_registry
from agentguard.core.redaction import sanitize_tool_call
from agentguard.core.trace import TraceStore, _register_active_recorder, _unregister_active_recorder, get_active_recorders
from agentguard.core.types import (
    BudgetConfig,
    CircuitBreakerConfig,
    GuardAction,
    GuardConfig,
    RateLimitConfig,
    RetryConfig,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    TraceEntry,
    ToolTimeoutError,
    ValidationResult,
    ValidatorKind,
)
from agentguard.guardrails.budget import BudgetState
from agentguard.guardrails.circuit_breaker import CircuitBreakerState
from agentguard.guardrails.shared import get_shared_budget, get_shared_circuit_breaker

F = TypeVar("F", bound=Callable[..., Any])


_rate_limiter_registry: dict[str, tuple["_RateLimiterState", RateLimitConfig]] = {}
_rate_limiter_registry_lock = threading.Lock()


def _resolve_rate_limit_key(tool_name: str, cfg: RateLimitConfig) -> Optional[str]:
    """Return the shared registry key for a rate limit config."""
    if cfg.shared_key == "":
        return None
    if cfg.shared_key is None:
        return tool_name
    return cfg.shared_key


def _rate_limit_configs_match(lhs: RateLimitConfig, rhs: RateLimitConfig) -> bool:
    """Compare effective rate limiter settings, excluding sharing metadata."""
    return (
        lhs.calls_per_second == rhs.calls_per_second
        and lhs.calls_per_minute == rhs.calls_per_minute
        and lhs.calls_per_hour == rhs.calls_per_hour
        and lhs.burst == rhs.burst
        and lhs.on_limit == rhs.on_limit
    )


def _get_or_create_rate_limiter_state(
    tool_name: str,
    cfg: RateLimitConfig,
) -> "_RateLimiterState":
    """Resolve shared rate limiter state for a tool/config pair."""
    shared_key = _resolve_rate_limit_key(tool_name, cfg)
    if shared_key is None:
        return _RateLimiterState(cfg)

    with _rate_limiter_registry_lock:
        existing = _rate_limiter_registry.get(shared_key)
        if existing is None:
            state = _RateLimiterState(cfg)
            _rate_limiter_registry[shared_key] = (state, cfg.model_copy(deep=True))
            return state

        state, existing_cfg = existing
        if not _rate_limit_configs_match(existing_cfg, cfg):
            warnings.warn(
                (
                    f"Conflicting rate limit config for shared key '{shared_key}'; "
                    "reusing the existing config and ignoring the new one."
                ),
                stacklevel=3,
            )
        return state


def _clear_rate_limiter_registry() -> None:
    """Clear shared rate limiter state. Intended for tests."""
    with _rate_limiter_registry_lock:
        _rate_limiter_registry.clear()


# ---------------------------------------------------------------------------
# Public factory / decorator
# ---------------------------------------------------------------------------


@overload
def guard(func: F) -> F: ...


@overload
def guard(
    func: None = None,
    *,
    config: Optional[GuardConfig] = None,
    validate_input: bool = False,
    validate_output: bool = False,
    detect_hallucination: bool = False,
    verify_response: bool = False,
    max_retries: int = 0,
    timeout: Optional[float] = None,
    budget: Optional[BudgetConfig] = None,
    circuit_breaker: Optional[CircuitBreakerConfig] = None,
    rate_limit: Optional[RateLimitConfig] = None,
    record: bool = False,
    trace_dir: str = "./traces",
    trace_backend: str = "sqlite",
    trace_db_path: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    registry: Any = None,
    before_call: Optional[Callable[..., None]] = None,
    after_call: Optional[Callable[..., None]] = None,
    custom_validators: Optional[list[Any]] = None,
    verification_engine: Optional[Any] = None,
    middleware: Any = None,
    shared_budget: Any = None,
    shared_circuit_breaker: Any = None,
) -> Callable[[F], F]: ...


def guard(  # type: ignore[misc]
    func: Optional[F] = None,
    *,
    config: Optional[GuardConfig] = None,
    validate_input: bool = False,
    validate_output: bool = False,
    detect_hallucination: bool = False,
    verify_response: bool = False,
    max_retries: int = 0,
    timeout: Optional[float] = None,
    budget: Optional[BudgetConfig] = None,
    circuit_breaker: Optional[CircuitBreakerConfig] = None,
    rate_limit: Optional[RateLimitConfig] = None,
    record: bool = False,
    trace_dir: str = "./traces",
    trace_backend: str = "sqlite",
    trace_db_path: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    registry: Any = None,
    before_call: Optional[Callable[..., None]] = None,
    after_call: Optional[Callable[..., None]] = None,
    custom_validators: Optional[list[Any]] = None,
    verification_engine: Optional[Any] = None,
    middleware: Any = None,
    shared_budget: Any = None,
    shared_circuit_breaker: Any = None,
) -> Any:
    """Decorate a callable with agentguard protection.

    Can be used as ``@guard`` (zero-config) or ``@guard(...)`` with keyword
    arguments. All keyword arguments map directly to :class:`~agentguard.core.types.GuardConfig`
    fields; pass *config* to supply a pre-built :class:`~agentguard.core.types.GuardConfig`.

    Args:
        func: The callable to wrap (when used as ``@guard`` without parentheses).
        config: Pre-built GuardConfig. Keyword arguments take precedence over
            fields in *config* when both are provided.
        validate_input: Validate arguments against type hints.
        validate_output: Validate return value against declared return type.
        verify_response: Run response anomaly detection on every call. Checks
            whether the response violates the tool's expected contract: anomalous
            execution timing, missing required fields, pattern mismatches, and
            statistically unusual values. Does *not* detect LLM-level factual
            hallucination.
        detect_hallucination: Deprecated alias for ``verify_response``.
        max_retries: Number of retries on failure.
        timeout: Timeout in seconds.
        budget: Budget enforcement configuration.
        circuit_breaker: Circuit breaker configuration.
        rate_limit: Rate limiter configuration.
        record: Persist traces to disk.
        trace_dir: Directory for trace files.
        trace_backend: Trace persistence backend (``sqlite`` or ``jsonl``).
        trace_db_path: SQLite database path when using the SQLite backend.
        session_id: Session label for grouping traces.
        tags: Labels attached to the registry entry.
        registry: Override the default global registry.
        before_call: Hook called just before the tool executes.
        after_call: Hook called just after the tool executes.

    Returns:
        A wrapped callable (or decorator factory when called with keyword args).
    """
    # verify_response is the canonical name; detect_hallucination is the legacy alias
    _do_verify = verify_response or detect_hallucination
    if shared_budget is not None:
        budget = shared_budget.config
    if shared_circuit_breaker is not None:
        circuit_breaker = shared_circuit_breaker.config

    # Build a GuardConfig from kwargs, merging with any provided config
    effective_config = _build_config(
        base=config,
        validate_input=validate_input,
        validate_output=validate_output,
        detect_hallucination=_do_verify,
        max_retries=max_retries,
        timeout=timeout,
        budget=budget,
        circuit_breaker=circuit_breaker,
        rate_limit=rate_limit,
        record=record,
        trace_dir=trace_dir,
        trace_backend=trace_backend,
        trace_db_path=trace_db_path,
        session_id=session_id,
        before_call=before_call,
        after_call=after_call,
        middleware=middleware,
    )

    def decorator(fn: F) -> F:
        guarded = GuardedTool(
            fn,
            config=effective_config,
            tags=tags or [],
            registry=registry,
            verification_engine=verification_engine,
        )
        if custom_validators:
            for v in custom_validators:
                guarded.add_validator(v)
        return cast(F, guarded)

    if func is not None:
        # Called as @guard with no parentheses
        return decorator(func)
    # Called as @guard(...) with keyword arguments
    return decorator


# ---------------------------------------------------------------------------
# GuardedTool — the core wrapper object
# ---------------------------------------------------------------------------


class GuardedTool:
    """Wraps a callable with the full agentguard protection stack.

    This class is not usually instantiated directly; use the :func:`guard`
    decorator instead.

    The call pipeline:
    1. Build :class:`~agentguard.core.types.ToolCall` record.
    2. Check circuit breaker (raise if OPEN).
    3. Check rate limiter (raise if exceeded).
    4. Check budget (raise if exceeded).
    5. Run before_call hook.
    6. Validate inputs (if enabled).
    7. Execute the wrapped function (with timeout, retries).
    8. Detect hallucination (if enabled).
    9. Validate outputs (if enabled).
    10. Update circuit breaker / budget state.
    11. Run after_call hook.
    12. Write trace entry (if record=True or active recorders exist).
    13. Return value (or raise on unrecoverable failure).
    """

    def __init__(
        self,
        func: Callable[..., Any],
        config: GuardConfig,
        tags: Optional[list[str]] = None,
        registry: Any = None,
        verification_engine: Optional[Any] = None,
    ) -> None:
        self._func = func
        self._config = config
        self._name = func.__name__
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__wrapped__ = func
        functools.update_wrapper(self, func)

        # Lazy imports to avoid circular dependencies
        self._circuit_breaker_state: Optional[Any] = None
        self._rate_limiter_state: Optional[_RateLimiterState] = None
        self._budget_state: Optional[Any] = None
        self._trace_store: Optional[TraceStore] = None
        self._middleware = config.middleware

        self._lock = threading.Lock()

        # Initialise guardrail states
        if config.circuit_breaker:
            shared_cb = (
                get_shared_circuit_breaker(config.circuit_breaker.shared_id)
                if config.circuit_breaker.shared_id
                else None
            )
            if shared_cb is not None:
                shared_cb.register_tool(self._name)
                self._circuit_breaker_state = shared_cb
            else:
                self._circuit_breaker_state = CircuitBreakerState(config.circuit_breaker)
        if config.rate_limit:
            self._rate_limiter_state = _get_or_create_rate_limiter_state(
                self._name,
                config.rate_limit,
            )
        if config.budget:
            shared_budget = (
                get_shared_budget(config.budget.shared_id)
                if config.budget.shared_id
                else None
            )
            if shared_budget is not None:
                shared_budget.register_tool(self._name)
                self._budget_state = shared_budget
            else:
                self._budget_state = BudgetState(config.budget)
        if config.record:
            self._trace_store = TraceStore(
                directory=config.trace_dir,
                backend=config.trace_backend,
                db_path=config.trace_db_path,
            )

        # Persistent verification engine (Bayesian multi-signal, replaces old heuristics)
        if verification_engine is not None:
            # Use caller-supplied engine (full control mode)
            self._hallucination_detector = verification_engine
        else:
            from agentguard.verification.engine import VerificationEngine
            self._hallucination_detector = VerificationEngine()
        # Keep both names for maximum compatibility
        self._verification_engine = self._hallucination_detector

        # Semantic validators
        self._semantic_validators: list[Any] = []
        # Custom validators
        self._custom_validators: list[Any] = []

        # Register in the global (or supplied) registry
        _reg = registry if registry is not None else global_registry
        reg_entry = ToolRegistration(
            name=self._name,
            func=func,
            guarded_func=self,
            tags=tags or [],
        )
        _reg.register(reg_entry, overwrite=True)
        self._registry = _reg

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the guarded tool call synchronously."""
        call = ToolCall(
            tool_name=self._name,
            args=args,
            kwargs=kwargs,
            session_id=self._config.session_id,
        )

        result = self._run_sync_pipeline(call)

        # Record to trace store if configured or active recorders exist
        self._write_trace(call, result)

        # Surface exception or return value
        if result.status not in (
            ToolCallStatus.SUCCESS,
            ToolCallStatus.RETRIED,
        ):
            raise _reconstruct_exception(result)
        return result.return_value

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the guarded tool call asynchronously."""
        call = ToolCall(
            tool_name=self._name,
            args=args,
            kwargs=kwargs,
            session_id=self._config.session_id,
        )
        result = await self._run_async_pipeline(call)
        self._write_trace(call, result)
        if result.status not in (ToolCallStatus.SUCCESS, ToolCallStatus.RETRIED):
            raise _reconstruct_exception(result)
        return result.return_value

    def _run_sync_pipeline(self, call: ToolCall) -> ToolResult:
        if self._middleware is None:
            return self._execute_sync(call)

        from agentguard.core.middleware import MiddlewareContext

        async def _terminal(_: Any) -> ToolResult:
            return self._execute_sync(call)

        ctx = MiddlewareContext(
            tool_name=self._name,
            args=call.args,
            kwargs=call.kwargs,
            config=self._config,
            call_id=call.call_id,
        )
        return _run_coroutine_sync(self._middleware.run(ctx, _terminal))

    async def _run_async_pipeline(self, call: ToolCall) -> ToolResult:
        if self._middleware is None:
            return await self._execute_async(call)

        from agentguard.core.middleware import MiddlewareContext

        async def _terminal(_: Any) -> ToolResult:
            return await self._execute_async(call)

        ctx = MiddlewareContext(
            tool_name=self._name,
            args=call.args,
            kwargs=call.kwargs,
            config=self._config,
            call_id=call.call_id,
        )
        return await self._middleware.run(ctx, _terminal)

    # ------------------------------------------------------------------
    # Execution pipeline (sync)
    # ------------------------------------------------------------------

    def _execute_sync(self, call: ToolCall) -> ToolResult:
        cfg = self._config

        # --- Circuit breaker check ---
        if self._circuit_breaker_state:
            block, recovery_in = self._circuit_breaker_state.check()
            if block:
                if hasattr(self._circuit_breaker_state, "increment_blocked"):
                    self._circuit_breaker_state.increment_blocked()
                result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.CIRCUIT_OPEN,
                    exception=f"Circuit breaker OPEN; recovery in {recovery_in:.1f}s",
                    exception_type="CircuitOpenError",
                )
                if cfg.circuit_breaker and cfg.circuit_breaker.on_open == GuardAction.BLOCK:
                    return result
                # WARN or LOG — proceed anyway
                self._log_warning(f"Circuit breaker OPEN for '{self._name}'")

        # --- Rate limiter check ---
        if self._rate_limiter_state:
            allowed, retry_after = self._rate_limiter_state.consume()
            if not allowed:
                result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.RATE_LIMITED,
                    exception=f"Rate limit exceeded; retry after {retry_after:.2f}s",
                    exception_type="RateLimitError",
                )
                if cfg.rate_limit and cfg.rate_limit.on_limit == GuardAction.BLOCK:
                    return result

        # --- Budget check ---
        if self._budget_state:
            exceeded, detail = self._budget_state.check_pre_call()
            if exceeded:
                result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.BUDGET_EXCEEDED,
                    exception=detail,
                    exception_type="BudgetExceededError",
                )
                if cfg.budget and cfg.budget.on_exceed == GuardAction.BLOCK:
                    return result

        # --- Before-call hook ---
        if cfg.before_call:
            try:
                cfg.before_call(call)
            except Exception as exc:
                pass  # hooks should not break the call

        # --- Input validation ---
        validations: list[ValidationResult] = []
        if cfg.validate_input:
            vr = self._validate_input(call)
            validations.extend(vr)
            failed = [v for v in vr if not v.valid]
            if failed:
                from agentguard.core.types import ValidationError
                msg = "; ".join(v.message for v in failed)
                result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.VALIDATION_FAILED,
                    exception=msg,
                    exception_type="ValidationError",
                    validations=validations,
                )
                return result

        # --- Execute with retries & timeout ---
        retry_cfg = cfg.retry or (
            RetryConfig(max_retries=cfg.max_retries) if cfg.max_retries > 0 else None
        )
        timeout_sec = cfg.timeout

        result = self._run_with_resilience(call, retry_cfg, timeout_sec, validations)

        # --- Output validation ---
        if cfg.validate_output and result.succeeded:
            out_vr = self._validate_output(call, result.return_value)
            result.validations.extend(out_vr)
            failed = [v for v in out_vr if not v.valid]
            if failed:
                msg = "; ".join(v.message for v in failed)
                result.status = ToolCallStatus.VALIDATION_FAILED
                result.exception = msg

        # --- Custom validators ---
        if self._custom_validators and result.succeeded:
            from agentguard.validators.custom import run_custom_validators
            custom_call = ToolCall(
                tool_name=self._name, args=call.args, kwargs=call.kwargs,
                session_id=call.session_id, call_id=call.call_id,
            )
            custom_vr = run_custom_validators(self._custom_validators, custom_call, result.return_value)
            result.validations.extend(custom_vr)
            failed_custom = [v for v in custom_vr if not v.valid]
            if failed_custom:
                msg = "; ".join(v.message for v in failed_custom)
                result.status = ToolCallStatus.VALIDATION_FAILED
                result.exception = msg

        # --- Hallucination detection ---
        if cfg.detect_hallucination and result.succeeded:
            hall_result = self._detect_hallucination(call, result)
            result.hallucination = hall_result
            if hall_result.is_hallucinated:
                result.status = ToolCallStatus.HALLUCINATED
                result.exception = hall_result.reason

        # --- Budget update ---
        if self._budget_state:
            self._budget_state.record_call()
            cost = result.cost or (cfg.budget.cost_per_call if cfg.budget else None)
            if cost and result.succeeded:
                self._budget_state.record_spend(cost)

        # --- Circuit breaker update ---
        if self._circuit_breaker_state:
            if result.succeeded:
                self._circuit_breaker_state.record_success()
            else:
                self._circuit_breaker_state.record_failure()

        # --- Registry stats ---
        self._registry.increment_calls(self._name)
        if not result.succeeded:
            self._registry.increment_failures(self._name)

        # --- After-call hook ---
        if cfg.after_call:
            try:
                cfg.after_call(call, result)
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Execution pipeline (async)
    # ------------------------------------------------------------------

    async def _execute_async(self, call: ToolCall) -> ToolResult:
        """Run the async variant. Delegates sync work to _execute_sync for
        the guardrail checks, then wraps the actual coroutine call."""
        cfg = self._config

        # Run synchronous checks
        for check_fn in [
            lambda: self._circuit_check_only(call),
            lambda: self._rate_check_only(call),
            lambda: self._budget_check_only(call),
        ]:
            block_result = check_fn()
            if block_result is not None:
                return block_result

        if cfg.before_call:
            try:
                cfg.before_call(call)
            except Exception:
                pass

        validations: list[ValidationResult] = []
        if cfg.validate_input:
            vr = self._validate_input(call)
            validations.extend(vr)
            failed = [v for v in vr if not v.valid]
            if failed:
                msg = "; ".join(v.message for v in failed)
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.VALIDATION_FAILED,
                    exception=msg,
                    exception_type="ValidationError",
                    validations=validations,
                )

        retry_cfg = cfg.retry or (
            RetryConfig(max_retries=cfg.max_retries) if cfg.max_retries > 0 else None
        )
        result = await self._run_with_resilience_async(
            call, retry_cfg, cfg.timeout, validations
        )

        if cfg.validate_output and result.succeeded:
            out_vr = self._validate_output(call, result.return_value)
            result.validations.extend(out_vr)
            failed = [v for v in out_vr if not v.valid]
            if failed:
                msg = "; ".join(v.message for v in failed)
                result.status = ToolCallStatus.VALIDATION_FAILED
                result.exception = msg

        # --- Custom validators ---
        if self._custom_validators and result.succeeded:
            from agentguard.validators.custom import run_custom_validators
            custom_call = ToolCall(
                tool_name=self._name, args=call.args, kwargs=call.kwargs,
                session_id=call.session_id, call_id=call.call_id,
            )
            custom_vr = run_custom_validators(self._custom_validators, custom_call, result.return_value)
            result.validations.extend(custom_vr)
            failed_custom = [v for v in custom_vr if not v.valid]
            if failed_custom:
                msg = "; ".join(v.message for v in failed_custom)
                result.status = ToolCallStatus.VALIDATION_FAILED
                result.exception = msg

        if cfg.detect_hallucination and result.succeeded:
            hall = self._detect_hallucination(call, result)
            result.hallucination = hall
            if hall.is_hallucinated:
                result.status = ToolCallStatus.HALLUCINATED
                result.exception = hall.reason

        if self._budget_state:
            self._budget_state.record_call()
            cost = result.cost or (cfg.budget.cost_per_call if cfg.budget else None)
            if cost and result.succeeded:
                self._budget_state.record_spend(cost)

        if self._circuit_breaker_state:
            if result.succeeded:
                self._circuit_breaker_state.record_success()
            else:
                self._circuit_breaker_state.record_failure()

        self._registry.increment_calls(self._name)
        if not result.succeeded:
            self._registry.increment_failures(self._name)

        if cfg.after_call:
            try:
                cfg.after_call(call, result)
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Resilience helpers
    # ------------------------------------------------------------------

    def _run_with_resilience(
        self,
        call: ToolCall,
        retry_cfg: Optional[RetryConfig],
        timeout_sec: Optional[float],
        validations: list[ValidationResult],
    ) -> ToolResult:
        max_attempts = 1 + (retry_cfg.max_retries if retry_cfg else 0)
        last_result: Optional[ToolResult] = None

        for attempt in range(max_attempts):
            start = time.perf_counter()
            try:
                if timeout_sec:
                    return_value = _run_with_timeout(self._func, self._name, timeout_sec, call.args, call.kwargs)
                else:
                    return_value = self._func(*call.args, **call.kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                last_result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.SUCCESS,
                    return_value=return_value,
                    execution_time_ms=elapsed_ms,
                    retry_count=attempt,
                    validations=list(validations),
                )
                return last_result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                status = ToolCallStatus.FAILURE
                if isinstance(exc, ToolTimeoutError):
                    status = ToolCallStatus.TIMEOUT

                last_result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=status,
                    exception=str(exc),
                    exception_type=type(exc).__qualname__,
                    execution_time_ms=elapsed_ms,
                    retry_count=attempt,
                    validations=list(validations),
                )
                if attempt < max_attempts - 1 and retry_cfg:
                    delay = _compute_delay(retry_cfg, attempt)
                    time.sleep(delay)
                else:
                    return last_result

        return last_result  # type: ignore[return-value]

    async def _run_with_resilience_async(
        self,
        call: ToolCall,
        retry_cfg: Optional[RetryConfig],
        timeout_sec: Optional[float],
        validations: list[ValidationResult],
    ) -> ToolResult:
        max_attempts = 1 + (retry_cfg.max_retries if retry_cfg else 0)
        last_result: Optional[ToolResult] = None

        for attempt in range(max_attempts):
            start = time.perf_counter()
            try:
                coro = self._func(*call.args, **call.kwargs)
                if timeout_sec:
                    return_value = await asyncio.wait_for(coro, timeout=timeout_sec)
                else:
                    return_value = await coro
                elapsed_ms = (time.perf_counter() - start) * 1000
                return ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=ToolCallStatus.SUCCESS,
                    return_value=return_value,
                    execution_time_ms=elapsed_ms,
                    retry_count=attempt,
                    validations=list(validations),
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                status = (
                    ToolCallStatus.TIMEOUT
                    if isinstance(exc, (asyncio.TimeoutError, ToolTimeoutError))
                    else ToolCallStatus.FAILURE
                )
                last_result = ToolResult(
                    call_id=call.call_id,
                    tool_name=self._name,
                    status=status,
                    exception=str(exc),
                    exception_type=type(exc).__qualname__,
                    execution_time_ms=elapsed_ms,
                    retry_count=attempt,
                    validations=list(validations),
                )
                if attempt < max_attempts - 1 and retry_cfg:
                    delay = _compute_delay(retry_cfg, attempt)
                    await asyncio.sleep(delay)
                else:
                    return last_result

        return last_result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_input(self, call: ToolCall) -> list[ValidationResult]:
        try:
            from agentguard.validators.schema import validate_inputs
            return validate_inputs(self._func, call.args, call.kwargs)
        except Exception as exc:
            return [ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Input validation error: {exc}",
            )]

    def _validate_output(self, call: ToolCall, return_value: Any) -> list[ValidationResult]:
        try:
            from agentguard.validators.schema import validate_output
            return validate_output(self._func, return_value)
        except Exception as exc:
            return [ValidationResult(
                valid=False,
                kind=ValidatorKind.SCHEMA,
                message=f"Output validation error: {exc}",
            )]

    def _detect_hallucination(self, call: ToolCall, result: ToolResult) -> Any:
        try:
            return self._hallucination_detector.verify(
                tool_name=self._name,
                execution_time_ms=result.execution_time_ms,
                response=result.return_value,
            )
        except Exception as exc:
            from agentguard.core.types import HallucinationResult
            return HallucinationResult(is_hallucinated=False, confidence=0.0, reason=str(exc))

    # ------------------------------------------------------------------
    # Guard check helpers (for async path)
    # ------------------------------------------------------------------

    def _circuit_check_only(self, call: ToolCall) -> Optional[ToolResult]:
        if not self._circuit_breaker_state:
            return None
        block, recovery_in = self._circuit_breaker_state.check()
        if block:
            cfg = self._config
            if hasattr(self._circuit_breaker_state, "increment_blocked"):
                self._circuit_breaker_state.increment_blocked()
            result = ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                status=ToolCallStatus.CIRCUIT_OPEN,
                exception=f"Circuit breaker OPEN; recovery in {recovery_in:.1f}s",
                exception_type="CircuitOpenError",
            )
            if cfg.circuit_breaker and cfg.circuit_breaker.on_open == GuardAction.BLOCK:
                return result
        return None

    def _rate_check_only(self, call: ToolCall) -> Optional[ToolResult]:
        if not self._rate_limiter_state:
            return None
        allowed, retry_after = self._rate_limiter_state.consume()
        if not allowed:
            cfg = self._config
            result = ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                status=ToolCallStatus.RATE_LIMITED,
                exception=f"Rate limit exceeded; retry after {retry_after:.2f}s",
                exception_type="RateLimitError",
            )
            if cfg.rate_limit and cfg.rate_limit.on_limit == GuardAction.BLOCK:
                return result
        return None

    def _budget_check_only(self, call: ToolCall) -> Optional[ToolResult]:
        if not self._budget_state:
            return None
        exceeded, detail = self._budget_state.check_pre_call()
        if exceeded:
            cfg = self._config
            result = ToolResult(
                call_id=call.call_id,
                tool_name=self._name,
                status=ToolCallStatus.BUDGET_EXCEEDED,
                exception=detail,
                exception_type="BudgetExceededError",
            )
            if cfg.budget and cfg.budget.on_exceed == GuardAction.BLOCK:
                return result
        return None

    # ------------------------------------------------------------------
    # Trace I/O
    # ------------------------------------------------------------------

    def _write_trace(self, call: ToolCall, result: ToolResult) -> None:
        sanitized_call = sanitize_tool_call(call, extra_fields=self._config.redact_fields)
        entry = TraceEntry(call=sanitized_call, result=result)
        # Write to configured trace store
        if self._trace_store:
            self._trace_store.write(entry, session_id=self._config.session_id)
        # Also write to any active context-manager recorders
        for rec in get_active_recorders():
            rec.record(entry)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_warning(msg: str) -> None:
        import warnings
        warnings.warn(msg, stacklevel=4)

    @property
    def config(self) -> GuardConfig:
        """Return the GuardConfig for this tool."""
        return self._config

    @property
    def original_func(self) -> Callable[..., Any]:
        """Return the original unwrapped callable."""
        return self._func

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        if self._circuit_breaker_state:
            self._circuit_breaker_state.reset()

    def reset_budget(self) -> None:
        """Reset session spend to zero."""
        if self._budget_state:
            self._budget_state.reset()

    def register_response_profile(
        self,
        *,
        expected_latency_ms: tuple[float, float] = (50, 30000),
        required_fields: Optional[list[str]] = None,
        response_patterns: Optional[list[str]] = None,
        forbidden_fields: Optional[list[str]] = None,
    ) -> "GuardedTool":
        """Register a response verification profile for this tool.

        Tells the anomaly detector what *normal* responses look like so it
        can flag deviations — missing required fields, anomalous timing, or
        pattern mismatches.

        Args:
            expected_latency_ms: ``(min, max)`` realistic execution time in ms.
                Responses outside this range are flagged.
            required_fields: Keys that must appear in dict responses.
                A response missing any of these is flagged.
            response_patterns: Regex patterns that should match in real responses.
                A response matching none of them is flagged.
            forbidden_fields: Field names that should never appear in valid responses.

        Returns:
            Self for method chaining.

        Example::

            @guard(verify_response=True)
            def get_weather(city: str) -> dict:
                ...

            get_weather.register_response_profile(
                expected_latency_ms=(100, 5000),
                required_fields=["temperature", "humidity"],
            )
        """
        self._hallucination_detector.register_tool(
            self._name,
            expected_latency_ms=expected_latency_ms,
            required_fields=required_fields or [],
            response_patterns=response_patterns or [],
        )
        return self

    def register_hallucination_profile(
        self,
        *,
        expected_latency_ms: tuple[float, float] = (50, 30000),
        required_fields: Optional[list[str]] = None,
        response_patterns: Optional[list[str]] = None,
        forbidden_fields: Optional[list[str]] = None,
    ) -> "GuardedTool":
        """Deprecated alias for :meth:`register_response_profile`."""
        return self.register_response_profile(
            expected_latency_ms=expected_latency_ms,
            required_fields=required_fields,
            response_patterns=response_patterns,
            forbidden_fields=forbidden_fields,
        )

    def add_validator(self, validator: Any) -> "GuardedTool":
        """Add a custom or semantic validator to this guarded tool.

        Args:
            validator: A ``CustomValidator`` instance or callable.

        Returns:
            Self for method chaining.
        """
        self._custom_validators.append(validator)
        return self

    # ------------------------------------------------------------------
    # Async session context manager
    # ------------------------------------------------------------------

    def session(self, *, session_id: Optional[str] = None) -> "_GuardedToolSession":
        """Return an async context manager that pins a session_id for all calls.

        While inside the ``async with`` block every :meth:`acall` (or
        ``await tool(...)`` call) uses *session_id* and trace entries are
        collected in memory.  On exit the session is flushed and the
        collected entries are returned from ``__aexit__``.

        Args:
            session_id: Explicit session label.  Auto-generated if omitted.

        Returns:
            An :class:`_GuardedToolSession` async context manager.

        Example::

            async with guarded_tool.session(session_id="my-session") as tool:
                result1 = await tool("query1")
                result2 = await tool("query2")
            # session_id is restored and entries are available via
            # tool.session_entries
        """
        return _GuardedToolSession(self, session_id=session_id or str(uuid.uuid4()))

    def __repr__(self) -> str:
        return f"GuardedTool(name={self._name!r})"


# ---------------------------------------------------------------------------
# Async session context manager helper
# ---------------------------------------------------------------------------


class _GuardedToolSession:
    """Async context manager returned by :meth:`GuardedTool.session`.

    Within the ``async with`` block the wrapped tool uses a fixed
    *session_id* and all trace entries produced by the tool are collected
    into an in-memory list that can be retrieved via :attr:`session_entries`
    after the block exits.

    Example::

        async with my_tool.session(session_id="abc") as tool:
            await tool("hello")
            await tool("world")
        # my_tool.session_entries now holds the two entries
    """

    def __init__(self, tool: "GuardedTool", session_id: str) -> None:
        self._tool = tool
        self._session_id = session_id
        self._entries: list[Any] = []
        self._previous_session_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Async context manager protocol
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "GuardedTool":
        """Pin session_id and start collecting trace entries."""
        # Save the previous session_id so we can restore it on exit
        self._previous_session_id = self._tool._config.session_id
        # Patch the live config object to use our session_id
        object.__setattr__(self._tool._config, "session_id", self._session_id)

        # Install a lightweight in-process recorder that just appends entries
        # to our local list (no file I/O needed)
        self._recorder = _InMemoryRecorder(self._entries)
        self._recorder.start()
        return self._tool

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Restore session_id and stop collecting."""
        self._recorder.stop()
        # Restore the original session_id
        object.__setattr__(self._tool._config, "session_id", self._previous_session_id)
        # Expose collected entries on the tool for easy post-block inspection
        self._tool._session_entries = list(self._entries)

    @property
    def entries(self) -> list[Any]:
        """Return trace entries collected so far (or after exit)."""
        return list(self._entries)


class _InMemoryRecorder:
    """Lightweight recorder that collects TraceEntry objects in a list.

    This is *not* a full :class:`TraceRecorder`; it only satisfies the
    interface that :func:`~agentguard.core.trace.get_active_recorders`
    returns (i.e. it has a ``record`` method) and registers/unregisters
    itself from the global active-recorder stack.
    """

    def __init__(self, target: list[Any]) -> None:
        self._target = target

    def start(self) -> None:
        _register_active_recorder(self)  # type: ignore[arg-type]

    def stop(self) -> None:
        _unregister_active_recorder(self)  # type: ignore[arg-type]

    def record(self, entry: Any) -> None:
        self._target.append(entry)


_CircuitBreakerState = CircuitBreakerState


class _RateLimiterState:
    """Token bucket rate limiter."""

    def __init__(self, cfg: RateLimitConfig) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._tokens = float(cfg.burst)
        self._last_refill = time.monotonic()
        # Determine tokens per second
        self._rate = 0.0
        if cfg.calls_per_second:
            self._rate = cfg.calls_per_second
        elif cfg.calls_per_minute:
            self._rate = cfg.calls_per_minute / 60.0
        elif cfg.calls_per_hour:
            self._rate = cfg.calls_per_hour / 3600.0

    def consume(self) -> tuple[bool, float]:
        """Try to consume one token. Returns (allowed, retry_after_seconds)."""
        if self._rate <= 0:
            return True, 0.0
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self._cfg.burst),
                self._tokens + elapsed * self._rate,
            )
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True, 0.0
            retry_after = (1.0 - self._tokens) / self._rate
            return False, retry_after


_BudgetState = BudgetState


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _build_config(
    base: Optional[GuardConfig],
    **kwargs: Any,
) -> GuardConfig:
    """Merge base config with kwargs overrides."""
    if base is None:
        base = GuardConfig()
    # Only override fields that differ from GuardConfig defaults
    update: dict[str, Any] = {}
    defaults = GuardConfig()
    for key, val in kwargs.items():
        if val != getattr(defaults, key, None):
            update[key] = val
    if update:
        return base.model_copy(update=update)
    return base


def _compute_delay(cfg: RetryConfig, attempt: int) -> float:
    """Compute the delay before the next retry attempt."""
    import random
    delay = min(cfg.initial_delay * (cfg.backoff_factor ** attempt), cfg.max_delay)
    if cfg.jitter:
        delay *= 0.5 + random.random()  # ±50% jitter (0.5x–1.5x)
    return delay


def _run_coroutine_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder: list[Any] = [None]
    error_holder: list[BaseException | None] = [None]

    def _runner() -> None:
        try:
            result_holder[0] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - passthrough
            error_holder[0] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error_holder[0] is not None:
        raise error_holder[0]
    return result_holder[0]


def _run_with_timeout(
    func: Callable[..., Any],
    tool_name: str,
    timeout_sec: float,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Execute *func* with a wall-clock timeout using a thread."""
    result_holder: list[Any] = [None, None]  # [return_value, exception]
    event = threading.Event()

    def _target() -> None:
        try:
            result_holder[0] = func(*args, **kwargs)
        except Exception as exc:
            result_holder[1] = exc
        finally:
            event.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    if not event.wait(timeout=timeout_sec):
        raise ToolTimeoutError(tool_name, timeout_sec)
    if result_holder[1] is not None:
        raise result_holder[1]
    return result_holder[0]


def _reconstruct_exception(result: ToolResult) -> Exception:
    """Reconstruct an exception from a failed ToolResult."""
    from agentguard.core.types import (
        BudgetExceededError,
        CircuitOpenError,
        HallucinationError,
        HallucinationResult,
        RateLimitError,
        ValidationError,
    )
    msg = result.exception or "Unknown error"
    exc_type = result.exception_type or ""
    if exc_type == "CircuitOpenError":
        err = CircuitOpenError(result.tool_name, 0.0)
        err.args = (msg,)  # Preserve original detail
        return err
    if exc_type == "BudgetExceededError":
        err = BudgetExceededError(result.tool_name, 0.0, 0.0)
        err.args = (msg,)
        return err
    if exc_type == "RateLimitError":
        err = RateLimitError(result.tool_name, 0.0)
        err.args = (msg,)
        return err
    if exc_type == "ValidationError":
        return ValidationError(msg)
    if exc_type == "HallucinationError" and result.hallucination:
        return HallucinationError(
            result.tool_name,
            result.hallucination,
        )
    if result.status == ToolCallStatus.TIMEOUT:
        err = ToolTimeoutError(result.tool_name, 0.0)
        err.args = (msg,)
        return err
    return RuntimeError(msg)
