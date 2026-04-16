"""agentguard — Runtime guardrails middleware for AI agent tool calls.

Quick start::

    from agentguard import guard

    @guard
    def search_web(query: str) -> dict:
        import requests
        return requests.get(f"https://api.search.com?q={query}").json()

With full configuration::

    from agentguard import guard, GuardConfig, CircuitBreaker, TokenBudget

    @guard(
        validate_input=True,
        validate_output=True,
        verify_response=True,   # detects anomalous responses (schema violations, timing, patterns)
        max_retries=3,
        timeout=30.0,
        budget=TokenBudget(max_cost_per_session=5.00).config,
        circuit_breaker=CircuitBreaker(failure_threshold=5).config,
        record=True,
    )
    def query_database(sql: str) -> list[dict]:
        ...

See the README for full documentation.
"""

from __future__ import annotations

# Core decorator and config
from agentguard.core.guard import GuardedTool, guard
from agentguard.core.middleware import (
    Middleware,
    MiddlewareChain,
    MiddlewareContext,
    NextFunc,
    logging_middleware,
    metadata_middleware,
    timing_middleware,
)
from agentguard.core.policy import (
    load_policy,
    apply_policy,
    validate_policy,
    PolicyError,
    PolicyValidationError,
)
from agentguard.core.registry import ToolRegistration, ToolRegistry, global_registry
from agentguard.core.trace import (
    JsonlTraceStore,
    SQLiteTraceStore,
    TraceRecorder,
    TraceStore,
    async_record_session,
    create_trace_store,
    record_session,
)
from agentguard.core.telemetry import StructuredLogger, instrument_agentguard
from agentguard.core.types import (
    AgentGuardError,
    AnomalousResponseError,
    BudgetConfig,
    BudgetExceededError,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    GuardAction,
    GuardConfig,
    HallucinationError,
    HallucinationResult,
    RateLimitConfig,
    RateLimitError,
    RetryConfig,
    TimeoutConfig,
    ToolTimeoutError,
    ToolCall,
    ToolCallStatus,
    ToolResult,
    TraceEntry,
    UsageKind,
    LLMUsage,
    LLMCostBreakdown,
    LLMSpendEvent,
    CostLedger,
    ValidationError,
    ValidationResult,
    ValidatorKind,
)

# Validators (legacy — kept for backward compatibility)
from agentguard.validators.hallucination import HallucinationDetector
from agentguard.validators.hallucination import ToolProfile as LegacyToolProfile

# Preferred alias with accurate name
ResponseVerifier = HallucinationDetector
from agentguard.validators.semantic import SemanticValidator
from agentguard.validators.custom import CustomValidator, validator_fn, no_empty_string_args

# Verification engine (new Bayesian multi-signal system)
from agentguard.verification import (
    VerificationEngine,
    VerificationResult,
    VerificationTier,
    SignalResult,
    ToolProfile,  # New ToolProfile from verification engine
    GLOBAL_LIKELIHOOD_RATIOS,
    ToolBaseline,
    RunningStats,
    SPCAnomaly,
    SPCResult,
    ConsistencyTracker,
    ConsistencyResult,
    ConsistencyViolation,
    AdaptiveThresholdManager,
    check_latency_anomaly,
    check_schema_compliance,
    check_response_patterns,
    check_response_length,
    check_value_plausibility,
    check_session_consistency,
)

# Guardrails — convenience classes
from agentguard.guardrails.circuit_breaker import CircuitBreaker
from agentguard.guardrails.rate_limiter import RateLimiter
from agentguard.guardrails.budget import TokenBudget
from agentguard.guardrails.retry import RetryPolicy, retry
from agentguard.guardrails.timeout import timeout

# Testing
from agentguard.testing.generator import TestGenerator
from agentguard.testing.assertions import assert_tool_call, AssertionBuilder
from agentguard.testing.replayer import TraceReplayer

# Reporting
from agentguard.reporting.console import ConsoleReporter
from agentguard.reporting.json_report import JsonReporter
from agentguard.costs import InMemoryCostLedger, NullCostLedger

# Multi-agent shared state
from agentguard.guardrails.shared import (
    SharedBudget,
    SharedCircuitBreaker,
    SharedBudgetStats,
    SharedCircuitStats,
    get_shared_budget,
    get_shared_circuit_breaker,
    clear_shared_registry,
)
from agentguard.integrations.tracked_clients import (
    guard_anthropic_client,
    guard_openai_client,
    guard_openai_compatible_client,
)

__version__ = "0.2.0"
__author__ = "Rigved Shirvalkar"
__license__ = "MIT"

__all__ = [
    # Decorator
    "guard",
    "GuardedTool",
    # Config
    "GuardConfig",
    "RetryConfig",
    "TimeoutConfig",
    "BudgetConfig",
    "CircuitBreakerConfig",
    "RateLimitConfig",
    # Types
    "ToolCall",
    "ToolResult",
    "TraceEntry",
    "UsageKind",
    "LLMUsage",
    "LLMCostBreakdown",
    "LLMSpendEvent",
    "CostLedger",
    "ValidationResult",
    "HallucinationResult",
    # Enums
    "GuardAction",
    "CircuitState",
    "ToolCallStatus",
    "ValidatorKind",
    # Errors
    "AgentGuardError",
    "ValidationError",
    "AnomalousResponseError",  # preferred name
    "HallucinationError",      # legacy alias
    "CircuitOpenError",
    "BudgetExceededError",
    "RateLimitError",
    "ToolTimeoutError",
    # Registry
    "ToolRegistry",
    "ToolRegistration",
    "global_registry",
    # Tracing
    "TraceRecorder",
    "TraceStore",
    "SQLiteTraceStore",
    "JsonlTraceStore",
    "create_trace_store",
    "record_session",
    "async_record_session",
    # Validators
    "ResponseVerifier",       # preferred name
    "HallucinationDetector",  # legacy alias
    "LegacyToolProfile",
    "SemanticValidator",
    "CustomValidator",
    "validator_fn",
    "no_empty_string_args",
    # Verification engine (new Bayesian multi-signal system)
    "VerificationEngine",
    "VerificationResult",
    "VerificationTier",
    "SignalResult",
    "ToolProfile",
    "GLOBAL_LIKELIHOOD_RATIOS",
    "ToolBaseline",
    "RunningStats",
    "SPCAnomaly",
    "SPCResult",
    "ConsistencyTracker",
    "ConsistencyResult",
    "ConsistencyViolation",
    "AdaptiveThresholdManager",
    "check_latency_anomaly",
    "check_schema_compliance",
    "check_response_patterns",
    "check_response_length",
    "check_value_plausibility",
    "check_session_consistency",
    # Guardrails
    "CircuitBreaker",
    "RateLimiter",
    "TokenBudget",
    "RetryPolicy",
    "retry",
    "timeout",
    # Testing
    "TestGenerator",
    "TraceReplayer",
    "assert_tool_call",
    "AssertionBuilder",
    # Reporting
    "ConsoleReporter",
    "JsonReporter",
    "InMemoryCostLedger",
    "NullCostLedger",
    "guard_openai_client",
    "guard_openai_compatible_client",
    "guard_anthropic_client",
    # Middleware
    "Middleware",
    "MiddlewareChain",
    "MiddlewareContext",
    "NextFunc",
    "logging_middleware",
    "metadata_middleware",
    "timing_middleware",
    # Policy-as-Code
    "load_policy",
    "apply_policy",
    "validate_policy",
    "PolicyError",
    "PolicyValidationError",
    # Shared state
    "SharedBudget",
    "SharedCircuitBreaker",
    "SharedBudgetStats",
    "SharedCircuitStats",
    "get_shared_budget",
    "get_shared_circuit_breaker",
    "clear_shared_registry",
    # Meta
    "__version__",
]
