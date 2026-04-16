"""Cross-session and within-session consistency checking for tool outputs.

Implements Section 7.4 "Cross-session consistency" from the research:
  "Track what tools return for similar queries across sessions. If
   get_stock_price(NVDA) returned $650 for the last 100 calls but now
   returns $50, flag for verification even if the individual output looks valid."
"""

from __future__ import annotations

import threading

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Consistency result
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyViolation:
    """Represents a detected consistency violation."""

    kind: str
    """'session' or 'historical'."""

    field: str
    """The field where the contradiction was found."""

    current_value: Any
    """The value in the current result."""

    prior_value: Any
    """The value in the conflicting prior result."""

    severity: float
    """How severe the inconsistency is (0.0–1.0)."""

    description: str


@dataclass
class ConsistencyResult:
    """Result of a consistency check."""

    is_consistent: bool
    violations: list[ConsistencyViolation] = field(default_factory=list)
    score: float = 0.0  # 0 = consistent, 1 = very inconsistent

    def __bool__(self) -> bool:
        return self.is_consistent


# ---------------------------------------------------------------------------
# ConsistencyTracker
# ---------------------------------------------------------------------------


class ConsistencyTracker:
    """Track tool outputs across calls for consistency checking.

    Maintains:
    - A session history (cleared when a new session starts).
    - A cross-session historical record per tool+args combination.

    Args:
        max_history_per_key: Maximum cross-session entries to keep per arg hash.
        session_window: Number of prior calls in the session to compare against.
        swing_threshold: Ratio change that constitutes an implausible swing.
    """

    def __init__(
        self,
        max_history_per_key: int = 50,
        session_window: int = 10,
        swing_threshold: float = 10.0,
    ) -> None:
        self.max_history_per_key = max_history_per_key
        self.session_window = session_window
        self.swing_threshold = swing_threshold

        # session_id -> tool_name -> list of {args, result}
        self._lock = threading.Lock()
        self._session_history: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # tool_name -> args_hash -> list of results (cross-session)
        self._cross_session_history: dict[str, dict[str, list[Any]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(
        self,
        tool_name: str,
        args: Any,
        result: Any,
        session_id: Optional[str] = None,
    ) -> None:
        """Record a completed tool call for future consistency checks.

        Args:
            tool_name: Name of the tool.
            args: Arguments passed to the tool.
            result: The result returned.
            session_id: Optional session identifier.
        """
        entry = {"args": args, "result": result}

        # Session history
        if session_id:
            history = self._session_history[session_id][tool_name]
            history.append(entry)
            # Trim to session_window * 2 to avoid unbounded growth
            if len(history) > self.session_window * 2:
                self._session_history[session_id][tool_name] = history[-(self.session_window * 2):]

        # Cross-session history (keyed by args hash for similar-input tracking)
        args_hash = _hash_args(args)
        cross_list = self._cross_session_history[tool_name][args_hash]
        cross_list.append(result)
        if len(cross_list) > self.max_history_per_key:
            self._cross_session_history[tool_name][args_hash] = cross_list[-self.max_history_per_key:]

    def check_session_consistency(
        self,
        tool_name: str,
        current_result: Any,
        session_id: Optional[str] = None,
    ) -> ConsistencyResult:
        """Detect whether the current result contradicts previous results in the session.

        Args:
            tool_name: Name of the tool.
            current_result: The result just produced.
            session_id: Session to check against.

        Returns:
            :class:`ConsistencyResult`.
        """
        if not session_id or not isinstance(current_result, dict):
            return ConsistencyResult(is_consistent=True)

        history = self._session_history.get(session_id, {}).get(tool_name, [])
        if not history:
            return ConsistencyResult(is_consistent=True)

        violations: list[ConsistencyViolation] = []
        recent = history[-self.session_window:]

        for prior_entry in recent:
            prior_result = prior_entry.get("result", {})
            if not isinstance(prior_result, dict):
                continue
            v = self._compare_results(current_result, prior_result, "session")
            violations.extend(v)

        if not violations:
            return ConsistencyResult(is_consistent=True)

        score = min(1.0, sum(v.severity for v in violations) / max(1, len(violations)))
        return ConsistencyResult(
            is_consistent=False,
            violations=violations,
            score=score,
        )

    def check_historical_consistency(
        self,
        tool_name: str,
        args: Any,
        current_result: Any,
    ) -> ConsistencyResult:
        """For similar inputs, check if the output matches historical patterns.

        If the same args have historically always returned a value in a certain
        range and this call is wildly different, flag it.

        Args:
            tool_name: Name of the tool.
            args: Arguments used in this call.
            current_result: The result just produced.

        Returns:
            :class:`ConsistencyResult`.
        """
        if not isinstance(current_result, dict):
            return ConsistencyResult(is_consistent=True)

        args_hash = _hash_args(args)
        history = self._cross_session_history.get(tool_name, {}).get(args_hash, [])

        if len(history) < 5:
            return ConsistencyResult(is_consistent=True)

        violations: list[ConsistencyViolation] = []
        # Compare against the last few historical results
        for prior_result in history[-5:]:
            if not isinstance(prior_result, dict):
                continue
            v = self._compare_results(current_result, prior_result, "historical")
            violations.extend(v)

        if not violations:
            return ConsistencyResult(is_consistent=True)

        score = min(1.0, sum(v.severity for v in violations) / max(1, len(violations)))
        return ConsistencyResult(
            is_consistent=False,
            violations=violations,
            score=score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compare_results(
        self,
        current: dict,
        prior: dict,
        kind: str,
    ) -> list[ConsistencyViolation]:
        """Compare two result dicts and return any violations."""
        violations = []
        for key, current_val in current.items():
            if key not in prior:
                continue
            prior_val = prior[key]

            # Both numeric: check for implausible swing
            if isinstance(current_val, (int, float)) and isinstance(prior_val, (int, float)):
                if prior_val == 0 and current_val == 0:
                    continue
                if prior_val == 0:
                    ratio = abs(current_val) + 1.0
                else:
                    ratio = abs(current_val / prior_val)

                # Flag extreme swings (> swing_threshold × change)
                if ratio > self.swing_threshold or (prior_val != 0 and ratio < 1.0 / self.swing_threshold):
                    severity = min(1.0, math.log10(max(ratio, 1.0 / max(ratio, 1e-10))) / 3.0)
                    violations.append(ConsistencyViolation(
                        kind=kind,
                        field=key,
                        current_value=current_val,
                        prior_value=prior_val,
                        severity=severity,
                        description=(
                            f"Field '{key}': {kind} inconsistency — "
                            f"was {prior_val}, now {current_val} "
                            f"({ratio:.1f}x change, threshold={self.swing_threshold}x)"
                        ),
                    ))

        return violations

    def clear_session(self, session_id: str) -> None:
        """Remove all session history for the given session_id."""
        self._session_history.pop(session_id, None)

    def get_session_history(self, tool_name: str, session_id: str) -> list[dict]:
        """Return session history for a tool."""
        return self._session_history.get(session_id, {}).get(tool_name, [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_args(args: Any) -> str:
    """Produce a stable hash for tool arguments (for cross-session lookup)."""
    try:
        serialised = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        serialised = str(args)
    return hashlib.md5(serialised.encode()).hexdigest()


# Math import at module level
import math
