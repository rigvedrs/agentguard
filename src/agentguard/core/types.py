"""Core type definitions for agentguard.

This module defines all shared data structures used throughout the library:
ToolCall, ToolResult, TraceEntry, GuardConfig, and related enums.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from agentguard.costs.types import (
    CostLedger,
    LLMCostBreakdown,
    LLMSpendEvent,
    LLMUsage,
    UsageKind,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GuardAction(str, Enum):
    """Action to take when a guard condition is violated."""

    BLOCK = "block"
    """Raise an exception and abort the tool call."""

    WARN = "warn"
    """Log a warning but allow the tool call to proceed."""

    LOG = "log"
    """Silently record the violation without interruption."""


class CircuitState(str, Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"
    """Normal operation — requests pass through."""

    OPEN = "open"
    """Failure threshold exceeded — requests are blocked."""

    HALF_OPEN = "half_open"
    """Recovery probe — a single request is allowed to test the tool."""


class ToolCallStatus(str, Enum):
    """Outcome of a guarded tool call."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMITED = "rate_limited"
    VALIDATION_FAILED = "validation_failed"
    HALLUCINATED = "hallucinated"
    RETRIED = "retried"


class ValidatorKind(str, Enum):
    """Category of a validator."""

    SCHEMA = "schema"
    HALLUCINATION = "hallucination"
    RESPONSE_VERIFICATION = "response_verification"
    SEMANTIC = "semantic"
    CUSTOM = "custom"

    @classmethod
    def _missing_(cls, value: object) -> "ValidatorKind | None":
        if value == "verify_response":
            return cls.RESPONSE_VERIFICATION
        return None


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A record of a single tool invocation before execution."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this call instance."""

    tool_name: str
    """Name of the tool (function) being called."""

    args: tuple[Any, ...] = Field(default_factory=tuple)
    """Positional arguments passed to the tool."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to the tool."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    """UTC time when the call was initiated."""

    session_id: Optional[str] = None
    """Optional session identifier for grouping related calls."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary metadata attached by the caller."""

    model_config = {"arbitrary_types_allowed": True}


class ValidationResult(BaseModel):
    """Result of a single validator check."""

    valid: bool
    """Whether the validation passed."""

    kind: ValidatorKind
    """Which category of validator produced this result."""

    message: str = ""
    """Human-readable description of the validation outcome."""

    details: dict[str, Any] = Field(default_factory=dict)
    """Additional structured details about the validation result."""


class HallucinationResult(BaseModel):
    """Result of tool response anomaly detection analysis.

    Despite the legacy name, this detects anomalous *tool responses* — not
    LLM-level hallucination. It fires when a response violates the expected
    contract: suspiciously fast execution, missing required fields, pattern
    mismatches, or statistically unusual values.
    """

    is_hallucinated: bool
    """True if the response is anomalous (violates expected contract).

    .. deprecated::
        Use ``is_anomalous`` instead for clarity.
    """

    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence score that the response is anomalous (0 = looks normal, 1 = definitely anomalous)."""

    reason: str = ""
    """Human-readable explanation for the anomaly determination."""

    signals: dict[str, Any] = Field(default_factory=dict)
    """Raw signal values used in the determination (timing, schema, patterns, etc.)."""

    @property
    def is_anomalous(self) -> bool:
        """True if the response is anomalous (violates expected contract)."""
        return self.is_hallucinated


class ToolResult(BaseModel):
    """A record of a completed tool invocation."""

    call_id: str
    """Matches the ToolCall.call_id for this invocation."""

    tool_name: str
    """Name of the tool that was called."""

    status: ToolCallStatus
    """Outcome of the tool call."""

    return_value: Any = None
    """The value returned by the tool, or None on failure."""

    exception: Optional[str] = None
    """Exception message if the call raised an error."""

    exception_type: Optional[str] = None
    """Fully qualified exception class name."""

    execution_time_ms: float = 0.0
    """Wall-clock time taken to execute the tool, in milliseconds."""

    retry_count: int = 0
    """Number of retries performed before this result."""

    validations: list[ValidationResult] = Field(default_factory=list)
    """All validation results applied to this call."""

    hallucination: Optional[HallucinationResult] = None
    """Response anomaly detection result, if verification was performed."""

    cost: Optional[float] = None
    """Estimated monetary cost of this call, in USD."""

    provider: Optional[str] = None
    """Model provider associated with this result when tracking LLM spend."""

    model: Optional[str] = None
    """Model identifier associated with this result when tracking LLM spend."""

    usage: Optional[LLMUsage] = None
    """Normalized usage payload for tracked LLM calls."""

    cost_breakdown: Optional[LLMCostBreakdown] = None
    """Resolved cost breakdown for tracked LLM calls."""

    cost_estimated: bool = False
    """Whether the reported cost is a fallback estimate."""

    cost_known: bool = False
    """Whether pricing resolved to a concrete cost."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    """UTC time when the call completed."""

    model_config = {"arbitrary_types_allowed": True}

    @property
    def succeeded(self) -> bool:
        """Return True if the tool call completed without error."""
        return self.status == ToolCallStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """Return True if the tool call resulted in any error or block."""
        return self.status not in (ToolCallStatus.SUCCESS, ToolCallStatus.RETRIED)


class TraceEntry(BaseModel):
    """Combined record of a tool call and its result, stored in trace logs."""

    call: ToolCall
    result: ToolResult

    @property
    def call_id(self) -> str:
        """Shortcut to the shared call_id."""
        return self.call.call_id

    @property
    def tool_name(self) -> str:
        """Shortcut to the tool name."""
        return self.call.tool_name


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Configuration for automatic retry behaviour."""

    max_retries: int = 3
    """Maximum number of retry attempts."""

    initial_delay: float = 1.0
    """Initial delay between retries, in seconds."""

    max_delay: float = 60.0
    """Maximum delay between retries, in seconds."""

    backoff_factor: float = 2.0
    """Multiplier applied to the delay on each successive retry."""

    jitter: bool = True
    """Add random jitter to delays to avoid thundering herd."""

    retryable_exceptions: tuple[type[Exception], ...] = Field(default_factory=tuple)
    """Exception types that trigger a retry. Empty means retry on all exceptions."""

    model_config = {"arbitrary_types_allowed": True}


class TimeoutConfig(BaseModel):
    """Configuration for tool call timeout enforcement."""

    timeout_seconds: float
    """Maximum wall-clock time allowed for a single tool call."""

    on_timeout: GuardAction = GuardAction.BLOCK
    """Action to take when the timeout is exceeded."""


class BudgetConfig(BaseModel):
    """Configuration for cost/token budget enforcement."""

    max_cost_per_call: Optional[float] = None
    """Maximum cost in USD for a single tool call. None = unlimited."""

    max_cost_per_session: Optional[float] = None
    """Maximum total cost in USD for the current session. None = unlimited."""

    max_calls_per_session: Optional[int] = None
    """Maximum number of tool calls in the current session. None = unlimited."""

    alert_threshold: float = 0.80
    """Fraction of the budget at which to emit a warning (0–1)."""

    on_exceed: GuardAction = GuardAction.BLOCK
    """Action to take when a budget limit is exceeded."""

    cost_per_call: Optional[float] = None
    """Fixed cost to attribute to each call when dynamic pricing is unavailable."""

    use_dynamic_llm_costs: bool = True
    """Whether LLM integrations should attempt real dynamic pricing."""

    model_pricing_overrides: dict[str, tuple[float, float]] = Field(default_factory=dict)
    """Explicit per-model input/output prices in USD per 1M tokens."""

    record_llm_spend: bool = True
    """Whether wrapped LLM integrations should emit spend events."""

    cost_ledger: Optional[CostLedger] = None
    """Optional ledger used to persist spend events beyond in-memory budgets."""

    shared_id: Optional[str] = None
    """Registry key for sharing budget state across guarded tools."""

    model_config = {"arbitrary_types_allowed": True}


class CircuitBreakerConfig(BaseModel):
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5
    """Number of consecutive failures required to open the circuit."""

    recovery_timeout: float = 60.0
    """Seconds to wait in OPEN state before allowing a probe in HALF_OPEN."""

    success_threshold: int = 1
    """Consecutive successes in HALF_OPEN required to transition back to CLOSED."""

    on_open: GuardAction = GuardAction.BLOCK
    """Action when the circuit is OPEN and a call arrives."""

    shared_id: Optional[str] = None
    """Registry key for sharing circuit-breaker state across guarded tools."""


class RateLimitConfig(BaseModel):
    """Configuration for a token-bucket rate limiter."""

    calls_per_second: Optional[float] = None
    """Maximum calls per second. None = no per-second limit."""

    calls_per_minute: Optional[float] = None
    """Maximum calls per minute. None = no per-minute limit."""

    calls_per_hour: Optional[float] = None
    """Maximum calls per hour. None = no per-hour limit."""

    burst: int = 1
    """Maximum burst size (tokens in the bucket at start)."""

    on_limit: GuardAction = GuardAction.BLOCK
    """Action to take when the rate limit is exceeded."""

    shared_key: Optional[str] = None
    """Shared bucket key.

    ``None`` shares by tool name, ``""`` disables sharing (per-instance),
    and any other string shares across all tools using that key.
    """


class GuardConfig(BaseModel):
    """Full configuration for the @guard decorator."""

    # Validation
    validate_input: bool = False
    """Validate function arguments against type hints and Pydantic models."""

    validate_output: bool = False
    """Validate the return value against the declared return type."""

    detect_hallucination: bool = False
    """Run response anomaly detection on every tool call.

    Checks whether the response violates the tool's expected contract:
    anomalous execution timing, missing required fields, pattern mismatches,
    and statistically unusual values. Does *not* detect LLM-level factual
    hallucination.

    .. note::
        Also settable via the ``verify_response`` alias on :func:`guard`.
    """

    # Resilience
    max_retries: int = 0
    """Number of automatic retries on failure (0 = no retries)."""

    retry: Optional[RetryConfig] = None
    """Fine-grained retry configuration. Overrides max_retries when set."""

    timeout: Optional[float] = None
    """Timeout in seconds. None = no timeout."""

    timeout_config: Optional[TimeoutConfig] = None
    """Fine-grained timeout configuration. Overrides timeout when set."""

    # Budget and rate limiting
    budget: Optional[BudgetConfig] = None
    """Cost/call budget enforcement."""

    rate_limit: Optional[RateLimitConfig] = None
    """Rate limiting configuration."""

    # Circuit breaker
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    """Circuit breaker configuration."""

    # Tracing
    record: bool = False
    """Record every call to the trace store for replay and test generation."""

    trace_dir: str = "./traces"
    """Directory used for JSONL traces or as the parent directory for SQLite."""

    trace_backend: str = "sqlite"
    """Trace persistence backend: ``sqlite`` or ``jsonl``."""

    trace_db_path: Optional[str] = None
    """SQLite database path. Defaults under ``trace_dir`` when unset."""

    session_id: Optional[str] = None
    """Optional session identifier to group related calls."""

    redact_fields: tuple[str, ...] = Field(default_factory=tuple)
    """Additional field names whose values should be redacted in traces/logs."""

    # Custom validators / hooks
    custom_validators: list[Callable[..., ValidationResult]] = Field(default_factory=list)
    """List of custom validator callables."""

    before_call: Optional[Callable[[ToolCall], None]] = None
    """Hook called just before the tool executes."""

    after_call: Optional[Callable[[ToolCall, ToolResult], None]] = None
    """Hook called just after the tool executes (success or failure)."""

    middleware: Any = None
    """Optional middleware chain applied around guarded execution."""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------


class AgentGuardError(Exception):
    """Base class for all agentguard errors."""


class ValidationError(AgentGuardError):
    """Raised when input or output validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class HallucinationError(AgentGuardError):
    """Raised when a tool response is flagged as anomalous.

    Despite the legacy name, this indicates an *anomalous response* —
    the response violated the expected contract (missing fields, anomalous
    timing, pattern mismatch, etc.), not that the LLM fabricated a result.

    .. deprecated::
        Prefer catching ``AnomalousResponseError``, which is an alias.
    """

    def __init__(self, tool_name: str, result: HallucinationResult) -> None:
        super().__init__(
            f"Tool '{tool_name}' returned an anomalous response "
            f"(confidence={result.confidence:.2f}): {result.reason}"
        )
        self.tool_name = tool_name
        self.result = result


# Preferred alias with accurate name
AnomalousResponseError = HallucinationError


class CircuitOpenError(AgentGuardError):
    """Raised when the circuit breaker is open and blocks a call."""

    def __init__(self, tool_name: str, recovery_in: float) -> None:
        super().__init__(
            f"Circuit breaker for '{tool_name}' is OPEN. "
            f"Recovery in {recovery_in:.1f}s."
        )
        self.tool_name = tool_name
        self.recovery_in = recovery_in


class BudgetExceededError(AgentGuardError):
    """Raised when a call would exceed the configured budget."""

    def __init__(self, tool_name: str, spent: float, limit: float) -> None:
        super().__init__(
            f"Budget exceeded for '{tool_name}': "
            f"spent ${spent:.4f} of ${limit:.4f} limit."
        )
        self.tool_name = tool_name
        self.spent = spent
        self.limit = limit


class RateLimitError(AgentGuardError):
    """Raised when a call exceeds the configured rate limit."""

    def __init__(self, tool_name: str, retry_after: float) -> None:
        super().__init__(
            f"Rate limit exceeded for '{tool_name}'. "
            f"Retry after {retry_after:.2f}s."
        )
        self.tool_name = tool_name
        self.retry_after = retry_after


class ToolTimeoutError(AgentGuardError):
    """Raised when a tool call exceeds its configured timeout."""

    def __init__(self, tool_name: str, timeout: float) -> None:
        super().__init__(f"Tool '{tool_name}' timed out after {timeout:.1f}s.")
        self.tool_name = tool_name
        self.timeout = timeout
        self.timeout_seconds = timeout
