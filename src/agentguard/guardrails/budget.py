"""Token/cost budget enforcement for AI tool calls.

Tracks cumulative spend across a session and enforces hard or soft limits.
Supports per-call limits, per-session limits, and call-count limits.

Usage::

    from agentguard.guardrails.budget import TokenBudget
    from agentguard import guard

    budget = TokenBudget(
        max_cost_per_call=0.10,
        max_cost_per_session=5.00,
        alert_threshold=0.80,
        on_exceed="block",
    )

    @guard(budget=budget.config)
    def expensive_llm_call(prompt: str) -> str:
        ...

    # Check spend at any time
    print(f"Session spend: ${budget.session_spend:.4f}")
"""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass
from typing import Optional

from agentguard.core.types import BudgetConfig, BudgetExceededError, GuardAction


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------


@dataclass
class BudgetStats:
    """Snapshot of budget usage at a point in time."""

    session_spend: float
    session_calls: int
    max_cost_per_session: Optional[float]
    max_cost_per_call: Optional[float]
    max_calls_per_session: Optional[int]
    budget_utilisation: Optional[float]
    """Fraction of session budget consumed (0–1), or None if no session limit."""
    calls_remaining: Optional[int]
    """Calls remaining before call limit is hit, or None if no call limit."""


class BudgetState:
    """Thread-safe budget state machine shared by guarded and standalone APIs."""

    def __init__(self, cfg: BudgetConfig) -> None:
        self._cfg = cfg
        self._session_spend = 0.0
        self._session_calls = 0
        self._lock = threading.Lock()
        self._spend_history: list[tuple[float, float]] = []

    def check_pre_call(self) -> tuple[bool, str]:
        with self._lock:
            if (
                self._cfg.max_calls_per_session is not None
                and self._session_calls >= self._cfg.max_calls_per_session
            ):
                return True, (
                    f"Max calls per session ({self._cfg.max_calls_per_session}) exceeded"
                )
            if (
                self._cfg.max_cost_per_session is not None
                and self._session_spend >= self._cfg.max_cost_per_session
            ):
                return True, (
                    f"Session cost budget (${self._cfg.max_cost_per_session:.2f}) exceeded; "
                    f"spent ${self._session_spend:.4f}"
                )
            return False, ""

    def check(self, estimated_cost: Optional[float] = None) -> None:
        exceeded, reason = self.check_pre_call()
        if exceeded:
            self._handle_exceed(reason)

        if estimated_cost is None:
            return

        with self._lock:
            if (
                self._cfg.max_cost_per_call is not None
                and estimated_cost > self._cfg.max_cost_per_call
            ):
                self._handle_exceed(
                    f"Estimated call cost ${estimated_cost:.4f} exceeds per-call limit "
                    f"${self._cfg.max_cost_per_call:.4f}"
                )
            if (
                self._cfg.max_cost_per_session is not None
                and (self._session_spend + estimated_cost) > self._cfg.max_cost_per_session
            ):
                self._handle_exceed(
                    f"Projected session spend ${self._session_spend + estimated_cost:.4f} "
                    f"would exceed session limit ${self._cfg.max_cost_per_session:.4f}"
                )

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
                    f"[agentguard] Budget alert for '{tool_name}': "
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

    def stats(self) -> BudgetStats:
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
            return BudgetStats(
                session_spend=self._session_spend,
                session_calls=self._session_calls,
                max_cost_per_session=self._cfg.max_cost_per_session,
                max_cost_per_call=self._cfg.max_cost_per_call,
                max_calls_per_session=self._cfg.max_calls_per_session,
                budget_utilisation=utilisation,
                calls_remaining=calls_remaining,
            )

    def _handle_exceed(self, reason: str) -> None:
        if self._cfg.on_exceed == GuardAction.BLOCK:
            raise BudgetExceededError("(tool)", self._session_spend, self._cfg.max_cost_per_session or 0.0)
        if self._cfg.on_exceed == GuardAction.WARN:
            warnings.warn(f"[agentguard] Budget exceeded: {reason}", stacklevel=6)


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


class TokenBudget:
    """Budget enforcer for agentguard-protected tools."""

    def __init__(
        self,
        *,
        max_cost_per_call: Optional[float] = None,
        max_cost_per_session: Optional[float] = None,
        max_calls_per_session: Optional[int] = None,
        alert_threshold: float = 0.80,
        on_exceed: str | GuardAction = GuardAction.BLOCK,
        cost_per_call: Optional[float] = None,
    ) -> None:
        """Initialise the budget enforcer.

        Args:
            max_cost_per_call: Maximum cost per individual tool call (USD).
            max_cost_per_session: Maximum cumulative cost for the session (USD).
            max_calls_per_session: Maximum number of calls in the session.
            alert_threshold: Fraction of the session budget at which a warning
                is emitted (default 0.80 = 80%).
            on_exceed: Action when a limit is breached (``"block"``, ``"warn"``, ``"log"``).
            cost_per_call: Fixed cost to attribute per call when the actual cost
                is not available from the tool result.
        """
        action = GuardAction(on_exceed) if isinstance(on_exceed, str) else on_exceed
        self._cfg = BudgetConfig(
            max_cost_per_call=max_cost_per_call,
            max_cost_per_session=max_cost_per_session,
            max_calls_per_session=max_calls_per_session,
            alert_threshold=alert_threshold,
            on_exceed=action,
            cost_per_call=cost_per_call,
        )
        self._state = BudgetState(self._cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, estimated_cost: Optional[float] = None) -> None:
        self._state.check(estimated_cost)

    def record_spend(
        self,
        cost: float,
        *,
        tool_name: str = "unknown",
    ) -> None:
        """Record a completed call's cost against the session budget.

        Args:
            cost: The actual cost of the call (USD).
            tool_name: Tool name for logging/alerting purposes.
        """
        self._state.record_call()
        self._state.record_spend(cost, tool_name=tool_name)

    def reset(self) -> None:
        """Reset the session spend and call count to zero."""
        self._state.reset()

    @property
    def session_spend(self) -> float:
        """Return the total spend in the current session (USD)."""
        return self._state.session_spend

    @property
    def session_calls(self) -> int:
        """Return the number of calls made in the current session."""
        return self._state.session_calls

    def stats(self) -> BudgetStats:
        """Return a snapshot of current budget utilisation."""
        return self._state.stats()

    @property
    def config(self) -> BudgetConfig:
        """Return the :class:`~agentguard.core.types.BudgetConfig`."""
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"TokenBudget("
            f"session=${self.session_spend:.4f}/"
            f"{self._cfg.max_cost_per_session or '∞'}, "
            f"calls={self.session_calls})"
        )
