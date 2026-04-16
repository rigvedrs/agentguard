"""Adaptive threshold management that learns per-tool thresholds from feedback.

Implements Section 4.4 "Adaptive Thresholds via Online Learning" from the research:
  "Start with a conservative global threshold, then learn per-tool thresholds
   from production traffic."

Uses Exponential Moving Average (EMA) updates on likelihood ratios to adapt
to deployment environment, per Section 4.2.
"""

from __future__ import annotations

import threading

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# AdaptiveThresholdManager
# ---------------------------------------------------------------------------


@dataclass
class ThresholdStats:
    """Statistics for a single tool's threshold adaptation."""

    tool_name: str
    threshold: float
    total_feedback: int = 0
    hallucination_count: int = 0
    correct_count: int = 0
    # EMA of P(hallucination) based on feedback
    ema_hallucination_rate: float = 0.15  # start from prior


class AdaptiveThresholdManager:
    """Learns per-tool thresholds from labelled feedback.

    When users or downstream systems flag a result as correct or hallucinated,
    this manager updates the threshold for that tool using EMA.

    Args:
        global_threshold: Default threshold for tools with no feedback yet.
        ema_alpha: EMA learning rate (0.0 = no learning, 1.0 = instant update).
        min_threshold: Floor threshold (never go below this).
        max_threshold: Ceiling threshold (never go above this).
    """

    def __init__(
        self,
        global_threshold: float = 0.5,
        ema_alpha: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
    ) -> None:
        if not 0.0 < global_threshold <= 1.0:
            raise ValueError("global_threshold must be in (0, 1]")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")

        self.global_threshold = global_threshold
        self._lock = threading.Lock()
        self.ema_alpha = ema_alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self._stats: dict[str, ThresholdStats] = {}
        # Per-signal likelihood ratios, per tool (overrides global)
        self._likelihood_ratios: dict[str, dict[str, float]] = defaultdict(dict)

    def get_threshold(self, tool_name: str) -> float:
        """Return the current threshold for a tool.

        Falls back to the global threshold for tools with no feedback.
        Thread-safe.

        Args:
            tool_name: Name of the tool.

        Returns:
            Threshold value in [min_threshold, max_threshold].
        """
        with self._lock:
            stats = self._stats.get(tool_name)
            if stats is None or stats.total_feedback == 0:
                return self.global_threshold
            return stats.threshold

    def record_feedback(
        self,
        tool_name: str,
        score: float,
        was_hallucination: bool,
    ) -> None:
        """Record a labelled feedback event and update the threshold.

        Args:
            tool_name: Name of the tool.
            score: The confidence score (0–1) the engine produced.
            was_hallucination: True if the call was actually hallucinated.
        """
        with self._lock:
            if tool_name not in self._stats:
                self._stats[tool_name] = ThresholdStats(
                    tool_name=tool_name,
                    threshold=self.global_threshold,
                )

            stats = self._stats[tool_name]
            stats.total_feedback += 1

            if was_hallucination:
                stats.hallucination_count += 1
            else:
                stats.correct_count += 1

            # EMA update of the hallucination rate
            label = 1.0 if was_hallucination else 0.0
            stats.ema_hallucination_rate = (
                (1.0 - self.ema_alpha) * stats.ema_hallucination_rate
                + self.ema_alpha * label
            )

            # Adapt threshold: if hallucination rate is high, lower threshold
            # (be stricter). 15% rate → 0.7 threshold.
            raw_threshold = 1.0 - 2.0 * stats.ema_hallucination_rate
            stats.threshold = max(
                self.min_threshold,
                min(self.max_threshold, raw_threshold),
            )

    def get_prior(self, tool_name: str) -> float:
        """Return the empirical hallucination prior for a tool.

        Falls back to the global prior of 0.15 if no feedback yet.

        Args:
            tool_name: Name of the tool.

        Returns:
            Estimated P(hallucination) for this tool.
        """
        with self._lock:
            stats = self._stats.get(tool_name)
            if stats is None or stats.total_feedback < 5:
                return 0.15  # Default from research section 4.2
            return stats.ema_hallucination_rate

    def update_likelihood_ratio(
        self,
        tool_name: str,
        signal_name: str,
        fired_hallucinated: float,
        fired_not_hallucinated: float,
        alpha: Optional[float] = None,
    ) -> None:
        """Online EMA update of a signal's likelihood ratio for a specific tool.

        Args:
            tool_name: Name of the tool.
            signal_name: Signal identifier.
            fired_hallucinated: New empirical P(signal fires | hallucination).
            fired_not_hallucinated: New empirical P(signal fires | not hallucination).
            alpha: Override global EMA alpha.
        """
        alpha = alpha or self.ema_alpha
        new_lr = fired_hallucinated / max(fired_not_hallucinated, 0.001)

        tool_ratios = self._likelihood_ratios[tool_name]
        if signal_name not in tool_ratios:
            # Import global defaults from engine
            from agentguard.verification.engine import GLOBAL_LIKELIHOOD_RATIOS
            tool_ratios[signal_name] = GLOBAL_LIKELIHOOD_RATIOS.get(signal_name, 3.0)

        tool_ratios[signal_name] = (
            (1.0 - alpha) * tool_ratios[signal_name]
            + alpha * new_lr
        )

    def get_likelihood_ratio(self, tool_name: str, signal_name: str) -> Optional[float]:
        """Return the per-tool likelihood ratio for a signal, or None for global default.

        Args:
            tool_name: Name of the tool.
            signal_name: Signal identifier.

        Returns:
            Likelihood ratio, or None if not yet adapted.
        """
        return self._likelihood_ratios.get(tool_name, {}).get(signal_name)

    def get_stats(self, tool_name: str) -> Optional[ThresholdStats]:
        """Return statistics for a tool."""
        return self._stats.get(tool_name)

    def reset(self, tool_name: Optional[str] = None) -> None:
        """Reset stats for a specific tool, or all tools if tool_name is None."""
        if tool_name:
            self._stats.pop(tool_name, None)
            self._likelihood_ratios.pop(tool_name, None)
        else:
            self._stats.clear()
            self._likelihood_ratios.clear()
