"""Circuit breaker pattern implementation.

A circuit breaker prevents a system from repeatedly calling a failing tool,
giving the downstream service time to recover.

State machine::

    CLOSED ──(failure_threshold reached)──▶ OPEN
      ▲                                        │
      │                                  (recovery_timeout elapsed)
      │                                        ▼
      └──(success_threshold reached)──── HALF_OPEN
                                         (single probe)

Usage::

    from agentguard.guardrails.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

    @guard(circuit_breaker=cb.config)
    def call_api(endpoint: str) -> dict:
        ...

    # Standalone usage
    try:
        cb.before_call("my_tool")
        result = call_api("/data")
        cb.after_success("my_tool")
    except Exception:
        cb.after_failure("my_tool")
        raise
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from agentguard.core.types import CircuitBreakerConfig, CircuitOpenError, CircuitState, GuardAction


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------


@dataclass
class CircuitStats:
    """Snapshot of circuit breaker statistics."""

    tool_name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    total_blocked: int
    opened_at: Optional[float]
    time_in_open_state: float


class CircuitBreakerState:
    """Thread-safe state for a single circuit breaker."""

    def __init__(self, cfg: CircuitBreakerConfig) -> None:
        self._cfg = cfg
        self.state: str = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.total_calls = 0
        self.total_blocked = 0
        self.opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def check(self) -> tuple[bool, float]:
        with self._lock:
            self.total_calls += 1
            if self.state == "closed":
                return False, 0.0
            if self.state == "open":
                elapsed = time.monotonic() - (self.opened_at or 0)
                remaining = max(0.0, self._cfg.recovery_timeout - elapsed)
                if elapsed >= self._cfg.recovery_timeout:
                    self.state = "half_open"
                    return False, 0.0
                return True, remaining
            return False, 0.0

    def record_success(self) -> None:
        with self._lock:
            if self.state == "half_open":
                self.success_count += 1
                if self.success_count >= self._cfg.success_threshold:
                    self.state = "closed"
                    self.failure_count = 0
                    self.success_count = 0
                    self.opened_at = None
            elif self.state == "closed":
                self.failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            if self.state in ("closed", "half_open"):
                if self.failure_count >= self._cfg.failure_threshold:
                    self.state = "open"
                    self.opened_at = time.monotonic()
                    self.success_count = 0

    def increment_blocked(self) -> None:
        with self._lock:
            self.total_blocked += 1

    def reset(self) -> None:
        with self._lock:
            self.state = "closed"
            self.failure_count = 0
            self.success_count = 0
            self.total_calls = 0
            self.total_blocked = 0
            self.opened_at = None


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Standalone circuit breaker that wraps any callable or integrates with @guard."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
        on_open: GuardAction = GuardAction.BLOCK,
        name: str = "default",
    ) -> None:
        """Initialise the circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening the circuit.
            recovery_timeout: Seconds to stay OPEN before probing.
            success_threshold: Consecutive successes in HALF_OPEN to close.
            on_open: Action when circuit is OPEN (BLOCK, WARN, or LOG).
            name: Human-readable identifier for this breaker.
        """
        self._cfg = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            on_open=on_open,
        )
        self.name = name

        # Per-tool state — keyed by tool_name
        self._states: dict[str, CircuitBreakerState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def before_call(self, tool_name: str) -> None:
        """Check the circuit before executing a tool call.

        Args:
            tool_name: The tool about to be called.

        Raises:
            CircuitOpenError: If the circuit is OPEN and on_open=BLOCK.
        """
        state = self._get_or_create(tool_name)
        blocked, recovery_in = state.check()
        if blocked:
            state.increment_blocked()
            if self._cfg.on_open == GuardAction.BLOCK:
                raise CircuitOpenError(tool_name, recovery_in)
            if self._cfg.on_open == GuardAction.WARN:
                import warnings
                warnings.warn(
                    f"Circuit breaker OPEN for '{tool_name}'; "
                    f"recovery in {recovery_in:.1f}s",
                    stacklevel=3,
                )

    def after_success(self, tool_name: str) -> None:
        """Record a successful call.

        Args:
            tool_name: The tool that succeeded.
        """
        self._get_or_create(tool_name).record_success()

    def after_failure(self, tool_name: str) -> None:
        """Record a failed call.

        Args:
            tool_name: The tool that failed.
        """
        self._get_or_create(tool_name).record_failure()

    def reset(self, tool_name: Optional[str] = None) -> None:
        """Reset the circuit breaker to CLOSED state.

        Args:
            tool_name: Specific tool to reset, or None to reset all.
        """
        with self._lock:
            if tool_name:
                if tool_name in self._states:
                    self._states[tool_name].reset()
            else:
                for s in self._states.values():
                    s.reset()

    def get_state(self, tool_name: str) -> CircuitState:
        """Return the current state for *tool_name*.

        Args:
            tool_name: The tool to query.

        Returns:
            One of CLOSED, OPEN, or HALF_OPEN.
        """
        return CircuitState(self._get_or_create(tool_name).state)

    def stats(self, tool_name: str) -> CircuitStats:
        """Return a statistics snapshot for *tool_name*.

        Args:
            tool_name: The tool to query.
        """
        s = self._get_or_create(tool_name)
        now = time.monotonic()
        return CircuitStats(
            tool_name=tool_name,
            state=CircuitState(s.state),
            failure_count=s.failure_count,
            success_count=s.success_count,
            total_calls=s.total_calls,
            total_blocked=s.total_blocked,
            opened_at=s.opened_at,
            time_in_open_state=(
                now - s.opened_at if s.opened_at and s.state == "open" else 0.0
            ),
        )

    @property
    def config(self) -> CircuitBreakerConfig:
        """Return the :class:`~agentguard.core.types.CircuitBreakerConfig`."""
        return self._cfg

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, tool_name: str) -> "CircuitBreakerState":
        with self._lock:
            if tool_name not in self._states:
                self._states[tool_name] = CircuitBreakerState(self._cfg)
            return self._states[tool_name]

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, "
            f"failure_threshold={self._cfg.failure_threshold}, "
            f"recovery_timeout={self._cfg.recovery_timeout})"
        )

