"""agentguard core package.

Re-exports the primary public API from the core submodules.
"""

from agentguard.core.guard import GuardedTool, guard
from agentguard.core.middleware import Middleware, MiddlewareChain, MiddlewareContext, NextFunc
from agentguard.core.policy import PolicyError, PolicyValidationError, apply_policy, load_policy, validate_policy
from agentguard.core.registry import ToolRegistration, ToolRegistry, global_registry
from agentguard.core.trace import (
    JsonlTraceStore,
    SQLiteTraceStore,
    TraceRecorder,
    TraceStore,
    create_trace_store,
    get_active_recorders,
    record_session,
)
from agentguard.core.telemetry import StructuredLogger, instrument_agentguard
from agentguard.core.types import (
    AgentGuardError,
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
    ValidationError,
    ValidationResult,
    ValidatorKind,
)

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
    "ValidationResult",
    "HallucinationResult",
    "Middleware",
    "MiddlewareChain",
    "MiddlewareContext",
    "NextFunc",
    # Enums
    "GuardAction",
    "CircuitState",
    "ToolCallStatus",
    "ValidatorKind",
    # Errors
    "AgentGuardError",
    "ValidationError",
    "HallucinationError",
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
    "get_active_recorders",
    "record_session",
    "load_policy",
    "apply_policy",
    "validate_policy",
    "PolicyError",
    "PolicyValidationError",
    "instrument_agentguard",
    "StructuredLogger",
]
