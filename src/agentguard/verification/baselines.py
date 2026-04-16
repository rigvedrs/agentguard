"""Per-tool statistical baselines using Statistical Process Control (SPC).

Implements running statistics and Western Electric SPC rules for detecting
anomalous tool behaviour over time.

References:
- Western Electric Handbook (1956) — the four SPC rules implemented here
- Research Section 7.4: "Latency-as-proof" — rigorous statistical model of
  expected latency distributions per tool.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Running statistics (Welford online algorithm)
# ---------------------------------------------------------------------------


class RunningStats:
    """Maintains running mean and variance using Welford's online algorithm.

    This avoids storing all values while providing accurate mean and std.
    A circular buffer of recent values is kept for SPC rule evaluation.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0  # Sum of squared deviations (Welford)
        self._window: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def update(self, value: float) -> None:
        """Add a new observation. Thread-safe."""
        with self._lock:
            self._count += 1
            self._window.append(value)
            # Welford update
            delta = value - self._mean
            self._mean += delta / self._count
            delta2 = value - self._mean
            self._M2 += delta * delta2

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def mean(self) -> float:
        with self._lock:
            return self._mean

    @property
    def variance(self) -> float:
        with self._lock:
            if self._count < 2:
                return 0.0
            return self._M2 / (self._count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def recent_values(self) -> list[float]:
        with self._lock:
            return list(self._window)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "count": self._count,
                "mean": self._mean,
                "std": math.sqrt(self._M2 / (self._count - 1)) if self._count >= 2 else 0.0,
                "min": min(self._window) if self._window else None,
                "max": max(self._window) if self._window else None,
            }


# ---------------------------------------------------------------------------
# SPC anomaly result
# ---------------------------------------------------------------------------


@dataclass
class SPCAnomaly:
    """A single SPC rule violation."""

    rule: str
    """Which Western Electric rule fired (e.g. 'rule_1_3sigma')."""

    field: str
    """Field name the violation was detected on."""

    value: float
    """The observed value that triggered the violation."""

    sigma_distance: float
    """How many standard deviations from the mean."""

    description: str
    """Human-readable description of the anomaly."""


@dataclass
class SPCResult:
    """Result of an SPC check against a tool baseline."""

    is_anomalous: bool
    anomalies: list[SPCAnomaly] = field(default_factory=list)
    score: float = 0.0  # 0 = normal, 1 = very anomalous

    def __bool__(self) -> bool:
        return self.is_anomalous


# ---------------------------------------------------------------------------
# ToolBaseline
# ---------------------------------------------------------------------------


class ToolBaseline:
    """Maintains running statistics for a tool's behaviour.

    Tracks latency, response size, field frequency, and numeric field ranges.
    Applies Western Electric SPC rules to detect anomalous responses.

    Args:
        tool_name: Name of the tool being tracked.
        window_size: Number of recent observations to keep in memory.
    """

    def __init__(self, tool_name: str, window_size: int = 100) -> None:
        self.tool_name = tool_name
        self.window_size = window_size
        self._lock = threading.Lock()
        self.latency_stats = RunningStats(window_size)
        self.response_size_stats = RunningStats(window_size)
        self.field_frequency: dict[str, int] = {}  # How often each field appears
        self.value_ranges: dict[str, RunningStats] = {}  # Stats per numeric field
        self.call_count = 0

    def record(
        self,
        execution_time_ms: float,
        response: Any,
        args: Optional[dict] = None,
    ) -> None:
        """Record a new data point.

        Args:
            execution_time_ms: Wall-clock execution time in ms.
            response: The tool response.
            args: Arguments passed to the tool (unused currently, reserved).
        """
        self.call_count += 1
        self.latency_stats.update(execution_time_ms)

        # Measure response size
        try:
            size = len(json.dumps(response, default=str))
        except Exception:
            size = 0
        self.response_size_stats.update(float(size))

        # Track field presence and numeric ranges
        if isinstance(response, dict):
            for key, val in response.items():
                self.field_frequency[key] = self.field_frequency.get(key, 0) + 1
                if isinstance(val, (int, float)):
                    if key not in self.value_ranges:
                        self.value_ranges[key] = RunningStats(self.window_size)
                    self.value_ranges[key].update(float(val))

    def check_anomaly(
        self,
        execution_time_ms: float,
        response: Any,
    ) -> SPCResult:
        """Run Western Electric SPC rules against the current data point.

        Requires at least 8 prior observations for meaningful SPC checks.

        Western Electric rules applied:
        1. 1 point beyond 3σ from mean
        2. 2 of 3 consecutive points beyond 2σ (same side)
        3. 4 of 5 consecutive points beyond 1σ (same side)
        4. 8 consecutive points on same side of mean

        Args:
            execution_time_ms: Execution time to check.
            response: Response to check.

        Returns:
            :class:`SPCResult` with any detected anomalies.
        """
        if self.call_count < 8:
            return SPCResult(is_anomalous=False)

        anomalies: list[SPCAnomaly] = []

        # --- Check latency ---
        latency_anomalies = self._apply_spc_rules(
            self.latency_stats, execution_time_ms, "latency_ms"
        )
        anomalies.extend(latency_anomalies)

        # --- Check response size ---
        try:
            size = float(len(json.dumps(response, default=str)))
        except Exception:
            size = 0.0

        size_anomalies = self._apply_spc_rules(
            self.response_size_stats, size, "response_size_bytes"
        )
        anomalies.extend(size_anomalies)

        # --- Check numeric fields ---
        if isinstance(response, dict):
            for key, val in response.items():
                if isinstance(val, (int, float)) and key in self.value_ranges:
                    stats = self.value_ranges[key]
                    if stats.count >= 8:
                        field_anomalies = self._apply_spc_rules(stats, float(val), key)
                        anomalies.extend(field_anomalies)

        # Score: each anomaly contributes, with rule 1 (extreme outlier) weighted most
        score = 0.0
        for a in anomalies:
            if "rule_1" in a.rule:
                score += 0.4
            elif "rule_2" in a.rule:
                score += 0.25
            elif "rule_3" in a.rule:
                score += 0.2
            elif "rule_4" in a.rule:
                score += 0.15
        score = min(1.0, score)

        return SPCResult(
            is_anomalous=len(anomalies) > 0,
            anomalies=anomalies,
            score=score,
        )

    # ------------------------------------------------------------------
    # SPC rules implementation
    # ------------------------------------------------------------------

    def _apply_spc_rules(
        self,
        stats: RunningStats,
        current_value: float,
        field_name: str,
    ) -> list[SPCAnomaly]:
        """Apply Western Electric rules to a single value against stats."""
        mean = stats.mean
        std = stats.std
        anomalies: list[SPCAnomaly] = []

        if std == 0:
            return anomalies

        z = (current_value - mean) / std
        recent = stats.recent_values

        # Rule 1: 1 point > 3σ from mean
        if abs(z) > 3.0:
            anomalies.append(SPCAnomaly(
                rule="rule_1_3sigma",
                field=field_name,
                value=current_value,
                sigma_distance=abs(z),
                description=(
                    f"{field_name}={current_value:.2f} is {abs(z):.1f}σ from mean "
                    f"{mean:.2f} (Rule 1: >3σ)"
                ),
            ))

        # Need at least 3 recent values for rules 2-4
        if len(recent) < 3:
            return anomalies

        # Build z-score series for recent window (including current)
        all_values = recent + [current_value]
        z_series = [(v - mean) / std for v in all_values]

        # Rule 2: 2 of last 3 points > 2σ (same side)
        last3_z = z_series[-3:]
        pos_count = sum(1 for z_val in last3_z if z_val > 2.0)
        neg_count = sum(1 for z_val in last3_z if z_val < -2.0)
        if pos_count >= 2 or neg_count >= 2:
            anomalies.append(SPCAnomaly(
                rule="rule_2_2of3_2sigma",
                field=field_name,
                value=current_value,
                sigma_distance=abs(z),
                description=(
                    f"{field_name}: 2 of last 3 points beyond 2σ on same side (Rule 2)"
                ),
            ))

        # Rule 3: 4 of last 5 points > 1σ (same side)
        if len(z_series) >= 5:
            last5_z = z_series[-5:]
            pos_count5 = sum(1 for z_val in last5_z if z_val > 1.0)
            neg_count5 = sum(1 for z_val in last5_z if z_val < -1.0)
            if pos_count5 >= 4 or neg_count5 >= 4:
                anomalies.append(SPCAnomaly(
                    rule="rule_3_4of5_1sigma",
                    field=field_name,
                    value=current_value,
                    sigma_distance=abs(z),
                    description=(
                        f"{field_name}: 4 of last 5 points beyond 1σ on same side (Rule 3)"
                    ),
                ))

        # Rule 4: 8 consecutive points on same side of mean
        if len(z_series) >= 8:
            last8_z = z_series[-8:]
            all_pos = all(z_val > 0 for z_val in last8_z)
            all_neg = all(z_val < 0 for z_val in last8_z)
            if all_pos or all_neg:
                side = "above" if all_pos else "below"
                anomalies.append(SPCAnomaly(
                    rule="rule_4_8_consecutive",
                    field=field_name,
                    value=current_value,
                    sigma_distance=abs(z),
                    description=(
                        f"{field_name}: 8 consecutive points {side} mean (Rule 4: process shift)"
                    ),
                ))

        return anomalies

    def to_dict(self) -> dict:
        """Serialise baseline state for inspection."""
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "latency": self.latency_stats.to_dict(),
            "response_size": self.response_size_stats.to_dict(),
            "field_frequency": self.field_frequency,
            "value_ranges": {
                k: v.to_dict() for k, v in self.value_ranges.items()
            },
        }
