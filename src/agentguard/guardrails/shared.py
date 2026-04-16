"""Shared budget and circuit-breaker state for multiple guarded tools."""

from __future__ import annotations

import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Optional

from agentguard.core.types import BudgetConfig, CircuitBreakerConfig, CircuitState, GuardAction

__all__ = [
    "SharedBudget",
    "SharedCircuitBreaker",
    "SharedBudgetStats",
    "SharedCircuitStats",
    "get_shared_budget",
    "get_shared_circuit_breaker",
    "clear_shared_registry",
]

_shared_budget_registry: dict[str, "SharedBudget"] = {}
_shared_cb_registry: dict[str, "SharedCircuitBreaker"] = {}
_registry_lock = threading.Lock()


def get_shared_budget(shared_id: str) -> Optional["SharedBudget"]:
    with _registry_lock:
        return _shared_budget_registry.get(shared_id)


def get_shared_circuit_breaker(shared_id: str) -> Optional["SharedCircuitBreaker"]:
    with _registry_lock:
        return _shared_cb_registry.get(shared_id)


def clear_shared_registry() -> None:
    with _registry_lock:
        _shared_budget_registry.clear()
        _shared_cb_registry.clear()


@dataclass
class SharedBudgetStats:
    shared_id: str
    session_spend: float
    session_calls: int
    max_cost_per_session: Optional[float]
    max_calls_per_session: Optional[int]
    budget_utilisation: Optional[float]
    calls_remaining: Optional[int]
    registered_tools: list[str] = field(default_factory=list)


@dataclass
class SharedCircuitStats:
    shared_id: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    total_blocked: int
    opened_at: Optional[float]
    time_in_open_state: float
    registered_tools: list[str] = field(default_factory=list)


class SharedBudget:
    """Thread-safe budget pool shared across multiple guarded tools."""

    def __init__(
        self,
        *,
        max_cost_per_call: Optional[float] = None,
        max_cost_per_session: Optional[float] = None,
        max_calls_per_session: Optional[int] = None,
        alert_threshold: float = 0.80,
        on_exceed: str | GuardAction = GuardAction.BLOCK,
        cost_per_call: Optional[float] = None,
        shared_id: Optional[str] = None,
        name: str = "",
    ) -> None:
        action = GuardAction(on_exceed) if isinstance(on_exceed, str) else on_exceed
        self.shared_id = shared_id or f"budget-{uuid.uuid4().hex[:8]}"
        self.name = name or self.shared_id
        self._cfg = BudgetConfig(
            max_cost_per_call=max_cost_per_call,
            max_cost_per_session=max_cost_per_session,
            max_calls_per_session=max_calls_per_session,
            alert_threshold=alert_threshold,
            on_exceed=action,
            cost_per_call=cost_per_call,
            shared_id=self.shared_id,
        )
        self._session_spend = 0.0
        self._session_calls = 0
        self._spend_history: list[tuple[float, float]] = []
        self._registered_tools: list[str] = []
        self._lock = threading.Lock()

        with _registry_lock:
            _shared_budget_registry[self.shared_id] = self

    def register_tool(self, tool_name: str) -> None:
        with self._lock:
            if tool_name not in self._registered_tools:
                self._registered_tools.append(tool_name)

    def check_pre_call(self) -> tuple[bool, str]:
        with self._lock:
            if (
                self._cfg.max_calls_per_session is not None
                and self._session_calls >= self._cfg.max_calls_per_session
            ):
                return True, (
                    f"Max calls per session ({self._cfg.max_calls_per_session}) exceeded "
                    f"(shared budget '{self.name}')"
                )
            if (
                self._cfg.max_cost_per_session is not None
                and self._session_spend >= self._cfg.max_cost_per_session
            ):
                return True, (
                    f"Session cost budget (${self._cfg.max_cost_per_session:.2f}) exceeded; "
                    f"spent ${self._session_spend:.4f} (shared budget '{self.name}')"
                )
            return False, ""

    def record_call(self) -> None:
        with self._lock:
            self._session_calls += 1

    def record_spend(self, cost: float, *, tool_name: str = "unknown") -> None:
        with self._lock:
            self._session_spend += cost
            self._spend_history.append((time.monotonic(), cost))
            if (
                self._cfg.max_cost_per_session is not None
                and self._session_spend
                >= self._cfg.max_cost_per_session * self._cfg.alert_threshold
            ):
                pct = (self._session_spend / self._cfg.max_cost_per_session) * 100
                warnings.warn(
                    f"[agentguard] SharedBudget '{self.name}' alert for '{tool_name}': "
                    f"${self._session_spend:.4f} ({pct:.0f}%) of "
                    f"${self._cfg.max_cost_per_session:.2f} session budget consumed.",
                    stacklevel=5,
                )

    def reset(self) -> None:
        with self._lock:
            self._session_spend = 0.0
            self._session_calls = 0
            self._spend_history.clear()

    @property
    def session_spend(self) -> float:
        with self._lock:
            return self._session_spend

    @property
    def session_calls(self) -> int:
        with self._lock:
            return self._session_calls

    @property
    def config(self) -> BudgetConfig:
        return self._cfg

    def stats(self) -> SharedBudgetStats:
        with self._lock:
            utilisation = (
                self._session_spend / self._cfg.max_cost_per_session
                if self._cfg.max_cost_per_session
                else None
            )
            calls_remaining = (
                self._cfg.max_calls_per_session - self._session_calls
                if self._cfg.max_calls_per_session is not None
                else None
            )
            return SharedBudgetStats(
                shared_id=self.shared_id,
                session_spend=self._session_spend,
                session_calls=self._session_calls,
                max_cost_per_session=self._cfg.max_cost_per_session,
                max_calls_per_session=self._cfg.max_calls_per_session,
                budget_utilisation=utilisation,
                calls_remaining=calls_remaining,
                registered_tools=list(self._registered_tools),
            )


class SharedCircuitBreaker:
    """Thread-safe circuit breaker shared across multiple guarded tools."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 1,
        on_open: GuardAction = GuardAction.BLOCK,
        shared_id: Optional[str] = None,
        name: str = "",
    ) -> None:
        self.shared_id = shared_id or f"cb-{uuid.uuid4().hex[:8]}"
        self.name = name or self.shared_id
        self._cfg = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            on_open=on_open,
            shared_id=self.shared_id,
        )
        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: Optional[float] = None
        self._total_calls = 0
        self._total_blocked = 0
        self._registered_tools: list[str] = []
        self._lock = threading.Lock()

        with _registry_lock:
            _shared_cb_registry[self.shared_id] = self

    def register_tool(self, tool_name: str) -> None:
        with self._lock:
            if tool_name not in self._registered_tools:
                self._registered_tools.append(tool_name)

    def check(self) -> tuple[bool, float]:
        with self._lock:
            self._total_calls += 1
            if self._state == "closed":
                return False, 0.0
            if self._state == "open":
                elapsed = time.monotonic() - (self._opened_at or 0.0)
                remaining = max(0.0, self._cfg.recovery_timeout - elapsed)
                if elapsed >= self._cfg.recovery_timeout:
                    self._state = "half_open"
                    return False, 0.0
                return True, remaining
            return False, 0.0

    def record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._success_count += 1
                if self._success_count >= self._cfg.success_threshold:
                    self._state = "closed"
                    self._failure_count = 0
                    self._success_count = 0
                    self._opened_at = None
            elif self._state == "closed":
                self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state in ("closed", "half_open"):
                if self._failure_count >= self._cfg.failure_threshold:
                    self._state = "open"
                    self._opened_at = time.monotonic()
                    self._success_count = 0

    def increment_blocked(self) -> None:
        with self._lock:
            self._total_blocked += 1

    def reset(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = None
            self._total_calls = 0
            self._total_blocked = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def circuit_state(self) -> CircuitState:
        return CircuitState(self.state)

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._cfg

    def stats(self) -> SharedCircuitStats:
        with self._lock:
            now = time.monotonic()
            return SharedCircuitStats(
                shared_id=self.shared_id,
                state=CircuitState(self._state),
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=self._total_calls,
                total_blocked=self._total_blocked,
                opened_at=self._opened_at,
                time_in_open_state=(
                    now - self._opened_at if self._opened_at and self._state == "open" else 0.0
                ),
                registered_tools=list(self._registered_tools),
            )
