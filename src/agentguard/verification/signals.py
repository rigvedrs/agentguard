"""Individual signal detectors for the VerificationEngine.

Each signal function returns a tuple of (fired: bool, score: float, detail: str).

- ``fired``: True if the signal indicates a response anomaly.
- ``score``: Anomaly strength in [0.0, 1.0]. 0 = response looks normal, 1 = definitely anomalous.
- ``detail``: Human-readable explanation of the anomaly.

These signals detect deviations from expected response contracts — not LLM-level
hallucination. Specifically:

- :func:`check_latency_anomaly`: flags responses that arrived impossibly fast
  (< 2ms), indicating no real I/O occurred (test stub left in production,
  cache misconfiguration, etc.).
- :func:`check_schema_compliance`: flags responses missing required fields or
  containing forbidden fields (API schema drift, partial responses).
- :func:`check_response_patterns`: flags responses that don't match expected
  regex patterns (error bodies returned as success).
- :func:`check_response_length`: flags responses that are unusually short or long.
- :func:`check_value_plausibility`: flags numeric values outside historical
  mean ± 3σ (unit changes, data corruption).
- :func:`check_session_consistency`: flags values that contradict earlier
  responses in the same session.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Signal: Latency anomaly
# ---------------------------------------------------------------------------


def check_latency_anomaly(
    execution_time_ms: float,
    expected_range: tuple[float, float],
    historical_stats: Optional[dict] = None,
) -> tuple[bool, float, str]:
    """Statistical latency check using z-score against historical distribution.

    For tools with network I/O, execution times < 2ms indicate no real I/O
    occurred — a test stub was left in production, a cache is short-circuiting
    the real call, or the function never reached the external system.
    Uses z-score against historical stats if available.

    Args:
        execution_time_ms: Observed execution time in milliseconds.
        expected_range: ``(min_ms, max_ms)`` plausible range.
        historical_stats: Optional dict with ``mean`` and ``std`` from baselines.

    Returns:
        (fired, score, detail)
    """
    min_ms, max_ms = expected_range

    # Universal: < 2ms almost certainly means no real I/O occurred
    if execution_time_ms < 2.0:
        score = 1.0 - (execution_time_ms / 2.0)
        return True, score, (
            f"Execution time {execution_time_ms:.2f}ms is below the 2ms minimum "
            f"for any real I/O operation — no network/disk call was made"
        )

    # Below expected minimum — but above the 2ms physical floor.
    # Use a gentler curve: score = (1 - ratio)^0.5 which is less aggressive
    # than linear. 5ms/50ms → 0.68 instead of 0.90; 25ms/50ms → 0.29.
    if execution_time_ms < min_ms:
        ratio = execution_time_ms / min_ms  # 0 < ratio < 1
        score = (1.0 - ratio) ** 0.5  # Square root: gentler curve
        detail = (
            f"Execution time {execution_time_ms:.2f}ms is below expected minimum "
            f"{min_ms:.0f}ms"
        )
        fired = score >= 0.3
        return fired, score, detail

    # Statistical z-score if we have history
    if historical_stats and historical_stats.get("std", 0) > 0:
        mean = historical_stats["mean"]
        std = historical_stats["std"]
        z = abs(execution_time_ms - mean) / std
        if z > 3.0:
            score = min(1.0, (z - 3.0) / 3.0)
            return True, score, (
                f"Execution time {execution_time_ms:.2f}ms is {z:.1f}σ from "
                f"historical mean {mean:.1f}ms"
            )

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Signal: Schema compliance
# ---------------------------------------------------------------------------


def check_schema_compliance(
    result: Any,
    required_fields: list[str],
    forbidden_fields: list[str],
) -> tuple[bool, float, str]:
    """Check required/forbidden fields in the response dict.

    Args:
        result: The tool result (should be a dict or JSON-serialisable).
        required_fields: Fields that must appear for the response to be valid.
        forbidden_fields: Fields that should never appear in a real response.

    Returns:
        (fired, score, detail)
    """
    if not required_fields and not forbidden_fields:
        return False, 0.0, ""

    # Normalise to dict
    if not isinstance(result, dict):
        try:
            as_json = json.dumps(result, default=str)
            result = json.loads(as_json)
        except Exception:
            pass

    if not isinstance(result, dict):
        if required_fields:
            return True, 0.8, (
                f"Response is not a dict but {len(required_fields)} required fields expected"
            )
        return False, 0.0, ""

    missing = [f for f in required_fields if f not in result]
    present_forbidden = [f for f in forbidden_fields if f in result]
    total = len(required_fields) + len(forbidden_fields)

    violations = len(missing) + len(present_forbidden)
    if violations == 0:
        return False, 0.0, ""

    score = violations / total
    parts: list[str] = []
    if missing:
        parts.append(f"Missing required fields: {missing}")
    if present_forbidden:
        parts.append(f"Forbidden fields present: {present_forbidden}")
    return True, score, "; ".join(parts)


# ---------------------------------------------------------------------------
# Signal: Response pattern matching
# ---------------------------------------------------------------------------


def check_response_patterns(
    result: Any,
    expected_patterns: list[str],
) -> tuple[bool, float, str]:
    """Regex pattern check: at least one pattern should match for real responses.

    Args:
        result: The tool result.
        expected_patterns: Regex patterns expected in a real response.

    Returns:
        (fired, score, detail)
    """
    if not expected_patterns:
        return False, 0.0, ""

    try:
        serialised = json.dumps(result, default=str)
    except Exception:
        serialised = str(result)

    for pat in expected_patterns:
        try:
            if re.search(pat, serialised):
                return False, 0.0, ""
        except re.error:
            continue

    return True, 0.8, f"Response matched none of {len(expected_patterns)} expected pattern(s)"


# ---------------------------------------------------------------------------
# Signal: Response length
# ---------------------------------------------------------------------------


def check_response_length(
    result: Any,
    min_length: Optional[int],
    max_length: Optional[int],
) -> tuple[bool, float, str]:
    """Response length bounds check.

    Args:
        result: The tool result.
        min_length: Minimum acceptable serialised length.
        max_length: Maximum acceptable serialised length.

    Returns:
        (fired, score, detail)
    """
    if min_length is None and max_length is None:
        return False, 0.0, ""

    try:
        length = len(json.dumps(result, default=str))
    except Exception:
        return False, 0.0, ""

    if min_length is not None and length < min_length:
        score = min(1.0, 0.5 * (1.0 - length / min_length))
        return True, score, f"Response length {length} < minimum {min_length}"

    if max_length is not None and length > max_length:
        return True, 0.2, f"Response length {length} > maximum {max_length}"

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Signal: Value plausibility (SPC)
# ---------------------------------------------------------------------------


def check_value_plausibility(
    result: Any,
    tool_name: str,
    historical_values: dict[str, list[float]],
) -> tuple[bool, float, str]:
    """Check if numeric values in result are within historical SPC ranges.

    For each numeric field, checks whether the observed value is within
    mean ± 3σ of historical observations.

    Args:
        result: The tool result (should be a dict).
        tool_name: Name of the tool (for logging).
        historical_values: Dict mapping field names to lists of past values.

    Returns:
        (fired, score, detail)
    """
    if not isinstance(result, dict) or not historical_values:
        return False, 0.0, ""

    anomalous_fields = []
    for field_name, past_values in historical_values.items():
        if field_name not in result:
            continue
        current = result[field_name]
        if not isinstance(current, (int, float)):
            continue
        if len(past_values) < 5:
            continue  # Not enough history

        mean = statistics.mean(past_values)
        try:
            std = statistics.stdev(past_values)
        except statistics.StatisticsError:
            continue

        if std == 0:
            continue

        z = abs(current - mean) / std
        if z > 3.0:
            anomalous_fields.append(f"{field_name}={current} ({z:.1f}σ from mean {mean:.2f})")

    if anomalous_fields:
        score = min(1.0, 0.3 * len(anomalous_fields))
        return True, score, f"SPC anomaly in fields: {'; '.join(anomalous_fields)}"

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Signal: Consistency within session
# ---------------------------------------------------------------------------


def check_session_consistency(
    tool_name: str,
    current_result: Any,
    session_history: list[dict],
    key_fields: Optional[list[str]] = None,
) -> tuple[bool, float, str]:
    """Detect whether the current result contradicts earlier results in the session.

    Args:
        tool_name: Name of the tool.
        current_result: The current tool result.
        session_history: List of previous ``{args, result}`` entries for this tool.
        key_fields: Fields to compare. If None, compares all numeric fields.

    Returns:
        (fired, score, detail)
    """
    if not isinstance(current_result, dict) or not session_history:
        return False, 0.0, ""

    contradictions = []
    for prior_entry in session_history[-5:]:  # Only check last 5
        prior_result = prior_entry.get("result", {})
        if not isinstance(prior_result, dict):
            continue
        for field_name, current_val in current_result.items():
            if not isinstance(current_val, (int, float, str)):
                continue
            if field_name not in prior_result:
                continue
            prior_val = prior_result[field_name]
            if type(current_val) != type(prior_val):
                continue
            # For numbers, flag if change is > 50x (implausible swing)
            if isinstance(current_val, (int, float)) and prior_val != 0:
                ratio = abs(current_val / prior_val)
                if ratio > 50 or ratio < 0.02:
                    contradictions.append(
                        f"{field_name}: was {prior_val}, now {current_val} ({ratio:.1f}x change)"
                    )

    if contradictions:
        score = min(1.0, 0.25 * len(contradictions))
        return True, score, f"Session contradiction: {'; '.join(contradictions)}"

    return False, 0.0, ""
